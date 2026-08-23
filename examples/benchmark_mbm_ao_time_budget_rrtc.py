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

from planners.anytime_gtmp import TensorState, ao_gtmp_plan
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

    # print(start, goals, valid)
    return start, goals, valid, problem_data


def _simplify_path_for_vis(path: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Drop consecutive duplicate waypoints for cleaner visualization scrubbing."""
    path = np.asarray(path)
    if path.ndim != 2 or len(path) <= 1:
        return path

    keep = [0]
    last = path[0]
    for i in range(1, len(path)):
        if not np.allclose(path[i], last, atol=atol, rtol=0.0):
            keep.append(i)
            last = path[i]

    if keep[-1] != len(path) - 1:
        keep.append(len(path) - 1)
    return path[keep]
    
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
        path = np.asarray(path)
        simplified_path = _simplify_path_for_vis(path)
        print(
            f"Visualizing simplified path: {len(simplified_path)} waypoints "
            f"(original: {len(path)})"
        )

        # Add a slider to scrub through the path
        slider = server.gui.add_slider(
            "Simplified Path Progress",
            min=0,
            max=len(simplified_path) - 1,
            step=1,
            initial_value=0,
        )

        @slider.on_update
        def _(_):
            idx = int(slider.value)
            if 0 <= idx < len(simplified_path):
                cfg = np.array(simplified_path[idx])
                urdf_vis.update_cfg(cfg)
    
    if not block:
        return server

    print("Viser running — press Ctrl+C to stop.")
    while True:
        time.sleep(0.1)


@hydra.main(version_base=None, config_path=get_configs_path().as_posix(), config_name="demo_gtmp_mbm")
def main(cfg: DictConfig):
    START_STATE, GOALS, VALID, PROBLEM_DATA = load_problem_globals(cfg)
    
    # Create VAMP environment from problem data
    env = vamp.problem_dict_to_vamp(PROBLEM_DATA)
    
    # Create VAMP-based cost function with RRTConnect validation
    # validation_method='rrtc' uses RRTConnect for more thorough connectivity checking
    # n_jobs: -1 = all CPUs, -2 = all but one, 1 = serial
    rrtc_max_iterations = 10
    print(f"Using RRTConnect validation with max_iterations={rrtc_max_iterations}")
    
    vamp_cost_fn = CostVAMP.create(
        env, 
        robot_name=cfg.robot, 
        n_jobs=cfg.get('n_jobs', -2),
        validation_method='rrtc',
        validation_max_iterations=rrtc_max_iterations
    )
    
    # Load robot for visualization
    urdf_path = get_data_path() / "panda/panda_spherized.urdf"
    urdf = yourdfpy.URDF.load(urdf_path.as_posix())
    ROBOT = Robot.from_urdf(urdf)
    
    bounds = jnp.stack([ROBOT.joints.lower_limits, ROBOT.joints.upper_limits], axis=1)
    
    # Use num_plans as batch_size to run multiple planning attempts in parallel
    batch_size = cfg.num_plans
        
    TENSOR_STATE = TensorState.create(
        dim=7,
        q=START_STATE,
        bounds=bounds,
        goals=GOALS,
        cost_fn=vamp_cost_fn,
        get_velocity=False,
        batch_size=batch_size,
        **cfg.planner.params
    )

    rng = jax.random.PRNGKey(0)
    
    # Use batch_size for parallel planning (gtmp_plan_anytime is already batched internally)
    print(f"Running with batch_size={batch_size}")
    
    print("Planning...")
    start_time = time.time()
    # gtmp_plan_anytime already handles batching internally, just call it once
    key = jax.random.PRNGKey(0)
    outputs, _ = ao_gtmp_plan(key, TENSOR_STATE, time_budget=120)
    # Block to ensure computation is done for timing
    outputs.path.block_until_ready()
    planning_time = time.time() - start_time
    print(f"Planning took {planning_time:.4f}s")
    
    paths = []
    # Move results to host
    collisions = jax.device_get(outputs.collision)
    computed_paths = jax.device_get(outputs.path)
    
    # collisions shape: (batch_size,), paths shape: (batch_size, num_layers+2, dim)
    for i in range(batch_size):
        if not collisions[i]:
            paths.append(computed_paths[i])

    print(f"\nResults:")
    print(f"  Validation method: RRTConnect (max_iterations={rrtc_max_iterations})")
    print(f"  Found {len(paths)} collision-free paths out of {batch_size} attempts")
    print(f"  Success rate: {len(paths) / batch_size * 100:.1f}%")
    print(f"  Total planning time: {planning_time:.4f}s")
    
    if len(paths) > 0:
        print(f"\nFirst path has {len(paths[0])} waypoints")
        obstacles = create_collision_environment(PROBLEM_DATA)
        visualize_path_with_viser(obstacles, paths[0], urdf=urdf, initial_config=START_STATE)
    else:
        print("No paths found!")


if __name__ == "__main__":
    main()
