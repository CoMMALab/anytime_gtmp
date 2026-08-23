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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils.costs._cost_functions import CostFinite, CostInfinite
from utils.obstacles import create_collision_environment, stack_obstacles

from planners.gtmp import TensorState, gtmp_plan
from planners.gtmp import *  


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


@hydra.main(version_base=None, config_path=get_configs_path().as_posix(), config_name="demo_gtmp_mbm")
def main(cfg: DictConfig):
    START_STATE, GOALS, VALID, PROBLEM_DATA = load_problem_globals(cfg)
    # rng_key = jax.random.PRNGKey(cfg.experiment.seed)
    DREAM_POINTS, PROBES = None, None
    urdf_path = get_data_path() / "panda/panda_spherized.urdf"
    urdf = yourdfpy.URDF.load(urdf_path.as_posix())
    ROBOT = Robot.from_urdf(urdf)
    ROBOT_COLL = RobotCollisionSpherized.from_urdf(urdf)
    STACKED_OBSTACLES = stack_obstacles(create_collision_environment(PROBLEM_DATA))

    # COST_FN = CostFinite(
    #         dim=7,
    #         robot=ROBOT,
    #         robot_coll=ROBOT_COLL,
    #         obstacles=STACKED_OBSTACLES,
    #         a=1.0,
    #         b=1.0,
    #         goal_pose=GOALS[0]
    #     )

    infinite_cost_fn = CostInfinite.create(
        robot=ROBOT,
        robot_coll=ROBOT_COLL,
        obstacles=STACKED_OBSTACLES,
        interpolation_steps=10, 
        cache_enabled=True  # Enable the new caching mechanism
    )

    bounds = jnp.stack([ROBOT.joints.lower_limits, ROBOT.joints.upper_limits], axis=1)
        
    TENSOR_STATE = TensorState.create(
        dim=7,
        q=START_STATE,
        bounds=bounds,
        goals=GOALS,
        occ_map=COST_FN,
        get_velocity=False,
        **cfg.planner.params
    )

    rng = jax.random.PRNGKey(0)
    
    # Use batch_size for parallel planning (gtmp_plan is already batched internally)
    batch_size = TENSOR_STATE.batch_size
    print(f"Running with batch_size={batch_size}")
    
    print("Planning...")
    start_time = time.time()
    # gtmp_plan already handles batching internally, just call it once
    key = jax.random.PRNGKey(0)
    outputs = gtmp_plan(key, TENSOR_STATE)
    # Block to ensure computation is done for timing
    outputs.path.block_until_ready()
    print(f"Planning took {time.time() - start_time:.4f}s")
    
    paths = []
    # Move results to host
    collisions = jax.device_get(outputs.collision)
    computed_paths = jax.device_get(outputs.path)
    
    for i in range(batch_size):
        if not collisions[i]:
            paths.append(computed_paths[i])

    print(f"Found {len(paths)} collision-free paths.")
    if len(paths) > 0:
        print(paths[0])
        obstacles = create_collision_environment(PROBLEM_DATA)
        visualize_path_with_viser(obstacles, paths[0], urdf=urdf, initial_config=START_STATE)
    else:
        print("No paths found!")






if __name__ == "__main__":
    main()
