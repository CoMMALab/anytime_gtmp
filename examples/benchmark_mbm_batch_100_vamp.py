import time
from pathlib import Path
import sys
import os
import pickle

import jax
import jax.numpy as jnp
from omegaconf import DictConfig
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
        validation_method='validate_motion',
    )

    urdf_path = get_data_path() / "panda/panda_spherized.urdf"
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
        num_layers=cfg.planner.params.get('num_layers', 30),
        num_dreams=cfg.planner.params.get('num_dreams', 200),
    )

    print("Running with",cfg.planner.params.get('num_layers', 30),"layers")
    print("Running with",cfg.planner.params.get('num_dreams', 200),"dreams")
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
    main()
