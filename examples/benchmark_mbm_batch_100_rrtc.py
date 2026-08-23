import time
from pathlib import Path
import sys
import os
import pickle

import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
import hydra

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from planners.anytime_gtmp import TensorState, gtmp_plan
from utils.costs._cost_functions import CostVAMP
from utils.obstacles import create_collision_environment
import vamp
import yourdfpy
from pyroffi import Robot

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

    problems = data['problems'][problem]
    problem_data = next(problem for problem in problems if problem['index'] == index)

    start = jnp.array(problem_data['start'])
    goals = jnp.array(problem_data['goals'])
    valid = problem_data['valid']
    return start, goals, valid, problem_data


@hydra.main(version_base=None, config_path=get_configs_path().as_posix(), config_name="demo_gtmp_mbm")
def main(cfg: DictConfig):
    START_STATE, GOALS, VALID, PROBLEM_DATA = load_problem_globals(cfg)

    env = vamp.problem_dict_to_vamp(PROBLEM_DATA)

    vamp_cost_fn = CostVAMP.create(
        env,
        robot_name=cfg.robot,
        n_jobs=cfg.get('n_jobs', -2),
        validation_method='rrtc',
        validation_max_iterations= 1_000_000_000
    )

    urdf_path = get_data_path() / cfg.robot / f"{cfg.robot}_spherized.urdf"
    urdf = yourdfpy.URDF.load(urdf_path.as_posix())
    ROBOT = Robot.from_urdf(urdf)

    bounds = jnp.stack([ROBOT.joints.lower_limits, ROBOT.joints.upper_limits], axis=1)

    batch_size = 100
    dim = bounds.shape[0]

    TENSOR_STATE = TensorState.create(
        dim=int(dim),
        q=START_STATE,
        bounds=bounds,
        goals=GOALS,
        cost_fn=vamp_cost_fn,
        get_velocity=False,
        batch_size=batch_size,
        num_layers=5,
        num_dreams=20,
    )

    print("Running with",5,"layers")
    print("Running with",20,"dreams")
    key = jax.random.PRNGKey(0)
    start_time = time.time()
    outputs = gtmp_plan(key, TENSOR_STATE)
    outputs.path.block_until_ready()
    planning_time = time.time() - start_time
    print(f"Planning took {planning_time:.4f}s (No warmup)")

    start_time = time.time()
    outputs = gtmp_plan(key, TENSOR_STATE)
    outputs.path.block_until_ready()
    planning_time = time.time() - start_time
    print(f"Planning took {planning_time:.4f}s (With warmup)")

    collisions = jax.device_get(outputs.collision)
    computed_paths = jax.device_get(outputs.path)
    success_count = int(jnp.sum(~collisions))
    print(f"Found {success_count} collision-free paths out of {batch_size} attempts")


if __name__ == '__main__':
    # robots = ['panda', 'fetch', 'ur5']
    # # problems = ['box', 'bookshelf_small', 'bookshelf_tall', 'bookshelf_thin', 'cage', 'table_pick', 'table_under_pick']
    # problems = ['box', 'table_under_pick', 'cage', 'bookshelf_small']

    robots = ['baxter']
    problems = ['bookshelf_tall_both_arms_easy']

    for robot in robots:
        for problem in problems:
            for idx in range(1, 2):  # 100 different problem instances
                
                cfg = {
                    'experiment': {'seed': 42},
                    'robot': robot,
                    'problem': problem,  # or specify a problem name
                    'index': idx,
                    'rrtc_max_iterations': 100000,
                    'n_jobs': -2,
                    'num_plans': 100,
                    'num_layers': 3,
                    'num_dreams': 20,
                    'min_layers': 1,
                    'max_layers': 5,
                    'min_dreams': 20,
                    'max_dreams': 100,
                    # 'time_budget': 120,
                    'planner': {
                        'params': {
                            # Add planner-specific params here
                        }
                    }
                }
                # Convert to OmegaConf DictConfig for compatibility
                cfg = OmegaConf.create(cfg)
                try:
                    main(cfg)
                except Exception as e:
                    print(f"Error processing {robot}/{problem}/{idx}: {e}")
                    continue
