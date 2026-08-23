import argparse
import pickle
import time
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from typing import Dict, List, Union
from pathlib import Path
import sys
import os
import yourdfpy
from pyroffi import Robot
from pyroffi.collision import RobotCollisionSpherized
import viser
from viser.extras import ViserUrdf
import vamp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils.costs._cost_functions import CostVAMP
from utils.obstacles import create_collision_environment, stack_obstacles

from planners.anytime_gtmp import TensorState, gtmp_plan_anytime
from planners.anytime_gtmp import *  

os.environ['JAX_PLATFORMS'] = 'cpu'
jax.config.update('jax_platform_name', 'cpu')

def get_configs_path():
    return Path(__file__).parent.parent / "configs"

def get_data_path():
    return Path(__file__).parent.parent / "data"

def load_problem_globals(cfg: DictConfig):
    rng_key = jax.random.PRNGKey(cfg.experiment.seed)
    robot = cfg.robot
    problem = cfg.problem
    index = cfg.index

    data_dir = get_data_path() / f"{robot}"
    with open(data_dir / "problems.pkl", 'rb') as f:
        data = pickle.load(f)

    if not problem:
        problem = list(data['problems'].keys())[0]

    if problem not in data['problems']:
        raise RuntimeError(
            f"""No problem with name {problem}!
                Existing problems: {list(data['problems'].keys())}"""
        )

    problems = data['problems'][problem]
    try:
        problem_data = next(problem for problem in problems if problem['index'] == index)
    except StopIteration:
        raise RuntimeError(f"No problem in {problem} with index {index}!")

    start = jnp.array(problem_data['start'])
    goals = jnp.array(problem_data['goals'])
    valid = problem_data['valid']

    return start, goals, valid, problem_data
    
def visualize_path_with_viser(obstacles, path, urdf=None, initial_config=None, block: bool = True):
    server = viser.ViserServer(host="0.0.0.0", port=8080)
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/world", show_axes=True)
    server.scene.add_grid("/ground", width=2, height=2)
    server.scene.add_light_hemisphere("/lights/ambient", intensity=0.6)
    server.scene.add_light_directional("/lights/key", intensity=2.0, cast_shadow=True)

    if urdf is not None:
        urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
        if initial_config is not None:
            urdf_vis.update_cfg(np.array(initial_config))
        ik_target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.5, 0.0, 0.5), wxyz=(0, 0, 1, 0)
    )

    for i, obs in enumerate(obstacles):
        if hasattr(obs, "to_trimesh"):
            mesh = obs.to_trimesh()
            server.scene.add_mesh_trimesh(name=f"/world/obstacles/obj_{i}", mesh=mesh, visible=True)
            continue

        # Unknown obstacle: place a small yellow icosphere at origin
        server.scene.add_icosphere(name=f"/world/obstacles/unknown_{i}", radius=0.05, position=(0, 0, 0), color=(255, 255, 0))

        
    print("Viser server started. Open the URL printed above to view obstacles.")
    
    if path is not None and len(path) > 0:
        # Add a slider to scrub through the path
        slider = server.gui.add_slider(
            "Path Progress",
            min=0,
            max=len(path) - 1,
            step=1,
            initial_value=0,
        )

        @slider.on_update
        def _(_):
            idx = int(slider.value)
            if 0 <= idx < len(path):
                cfg = np.array(path[idx])
                urdf_vis.update_cfg(cfg)
    
    if not block:
        return server

    print("Viser running — press Ctrl+C to stop.")
    while True:
        time.sleep(0.1)


def entropy_path(paths: np.ndarray) -> float:
    """Compute path diversity entropy using pairwise distances."""
    # Compute pairwise path distances (Frechet-like)
    frechet_matrix = np.linalg.norm(
        paths[:, None, :, :] - paths[None, :, :, :], axis=-1
    ).sum(axis=-1)
    # Normalize to create probability distribution
    frechet_matrix = frechet_matrix / (frechet_matrix.sum() + 1e-12)
    # Compute entropy
    entropy = -np.sum(frechet_matrix * np.log(frechet_matrix + 1e-12))
    return entropy


def compute_metrics(paths: np.ndarray, collisions: np.ndarray) -> dict:
    """Compute planning metrics: collision-free %, average path length, path entropy."""
    collision_free_pct = 1.0 - collisions.mean()
    
    free_paths = paths[~collisions]
    if free_paths.shape[0] == 0:
        return {
            'collision_free_pct': collision_free_pct,
            'avg_path_length': float('nan'),
            'path_entropy': float('nan')
        }
    
    # Compute average path length (sum of segment distances)
    path_lengths = np.linalg.norm(
        free_paths[:, 1:, :] - free_paths[:, :-1, :], axis=-1
    ).sum(axis=-1)
    avg_path_length = path_lengths.mean()
    
    # Compute path diversity entropy
    path_entropy = entropy_path(free_paths)
    
    return {
        'collision_free_pct': collision_free_pct,
        'avg_path_length': avg_path_length,
        'path_entropy': path_entropy
    }


@hydra.main(version_base=None, config_path=get_configs_path().as_posix(), config_name="demo_gtmp_anytime")
def main(cfg: DictConfig):
    START_STATE, GOALS, VALID, PROBLEM_DATA = load_problem_globals(cfg)
    
    # Create VAMP environment from problem data
    env = vamp.problem_dict_to_vamp(PROBLEM_DATA)
    
    # Create VAMP-based cost function with CPU parallelism for validation
    # n_jobs: -1 = all CPUs, -2 = all but one, 1 = serial
    vamp_cost_fn = CostVAMP.create(env, robot_name=cfg.robot, n_jobs=-2)
    
    # Load robot for visualization
    urdf_path = get_data_path() / "panda/panda_spherized.urdf"
    urdf = yourdfpy.URDF.load(urdf_path.as_posix())
    ROBOT = Robot.from_urdf(urdf)
    
    bounds = jnp.stack([ROBOT.joints.lower_limits, ROBOT.joints.upper_limits], axis=1)
    
    # Create state (num_layers and num_dreams will be overridden by anytime planner)
    TENSOR_STATE = TensorState.create(
        dim=7,
        q=START_STATE,
        bounds=bounds,
        goals=GOALS,
        cost_fn=vamp_cost_fn,
        get_velocity=False,
        batch_size=cfg.get('batch_size', 10),  # batch_size for parallel attempts
        num_dreams=20,  # Will be overridden
        num_layers=1,   # Will be overridden
        vi_finite=True
    )

    rng = jax.random.PRNGKey(cfg.experiment.seed)
    
    print(f"Running anytime planner with batch_size={TENSOR_STATE.batch_size}")
    print(f"Problem: {cfg.problem}, Index: {cfg.index}, Valid: {VALID}")
    print()
    
    # Run anytime planner
    start_time = time.time()
    output, _ = gtmp_plan_anytime(
        rng, 
        TENSOR_STATE,
        min_layers=cfg.anytime.get('min_layers', 1),
        max_layers=cfg.anytime.get('max_layers', 30),
        min_dreams=cfg.anytime.get('min_dreams', 20),
        max_dreams=cfg.anytime.get('max_dreams', 200),
        dreams_step=cfg.anytime.get('dreams_step', 10),
        verbose=True
    )
    planning_time = time.time() - start_time
    
    print()
    print(f"Planning took {planning_time:.4f}s total")
    
    # Extract collision-free paths
    collisions = jax.device_get(output.collision)
    computed_paths = jax.device_get(output.path)
    
    paths = []
    for i in range(len(collisions)):
        if not collisions[i]:
            paths.append(computed_paths[i])

    print(f"Found {len(paths)} collision-free paths.")
    
    # Compute metrics
    metrics = compute_metrics(computed_paths, collisions)
    print(f"\nMetrics:")
    print(f"  Collision-free percentage: {metrics['collision_free_pct']:.2f}")
    print(f"  Averaged path length: {metrics['avg_path_length']:.4f}")
    print(f"  Path entropy: {metrics['path_entropy']:.4f}")
    
    if len(paths) > 0:
        print(f"\nSolution details:")
        print(f"  Layers: {output.num_layers}")
        print(f"  Dreams: {output.num_dreams}")
        print(f"  Iterations: {output.iterations}")
        if output.timing:
            print(f"  Iteration time: {output.timing.get('iteration_time', 0):.4f}s")
            print(f"  Total search time: {output.timing.get('total_time', 0):.4f}s")
        print(f"\nFirst path:")
        print(paths[0])
        
        obstacles = create_collision_environment(PROBLEM_DATA)
        visualize_path_with_viser(obstacles, paths[0], urdf=urdf, initial_config=START_STATE)
    else:
        print("No paths found!")


if __name__ == "__main__":
    main()
