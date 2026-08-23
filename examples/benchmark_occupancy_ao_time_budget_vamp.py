import argparse
import time
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from PIL import Image as PILImage

import jax
import jax.numpy as jnp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils.costs._cost_functions import CostVAMP

from planners.anytime_gtmp import TensorState, ao_gtmp_plan
from planners.anytime_gtmp import *

from utils.costs._occupancy_map import build_environment

os.environ['JAX_PLATFORMS'] = 'cpu'
jax.config.update('jax_platform_name', 'cpu')


START_STATE = jnp.array([5.5, 4.9, 0.5])
GOAL_STATE = jnp.array([15.5, 16.2, 0.5])


def get_gallery_path():
    return Path(__file__).parent.parent / "gallery"


def _add_hyperellipsoid(ax, q_xy, goal_xy, c_best, c_min):
    """Overlay the informed RRT* cost-bound hyperellipsoid projected to xy."""
    if not np.isfinite(c_best) or c_best <= c_min:
        return
    a = c_best / 2.0
    b = float(np.sqrt(max(a * a - (c_min / 2.0) ** 2, 0.0)))
    center = (q_xy + goal_xy) / 2.0
    angle = float(np.degrees(np.arctan2(goal_xy[1] - q_xy[1], goal_xy[0] - q_xy[0])))
    ellipse = Ellipse(
        xy=center, width=2 * a, height=2 * b, angle=angle,
        fill=False, edgecolor='orange', linewidth=1.8, linestyle='--', zorder=5,
    )
    ax.add_patch(ellipse)


def save_ao_progress_gif(
    iter_best_paths, npy_path, output_path, q_3d, goal_3d,
    title="", fps=4, max_frames=40,
):
    """Animate AO-GTMP iterations: incumbent path + cost-bound hyperellipsoid per iter."""
    map_data = np.load(npy_path)
    entries = iter_best_paths[:max_frames]

    q_xy = np.asarray(q_3d[:2], dtype=float)
    goal_xy = np.asarray(goal_3d[:2], dtype=float)
    c_min = float(np.linalg.norm(np.asarray(goal_3d, dtype=float) - np.asarray(q_3d, dtype=float)))

    frames = []
    for i, entry in enumerate(entries):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
        ax.imshow(map_data.T, extent=[0, 20, 0, 20],
                  origin='lower', cmap='gray', alpha=0.65, aspect='auto')

        c_best = float(entry['c_best'])
        _add_hyperellipsoid(ax, q_xy, goal_xy, c_best, c_min)

        for prior in entries[:i + 1]:
            prior_path = prior.get('iter_best_path')
            if prior_path is None:
                continue
            pts = np.asarray(prior_path)
            ax.plot(pts[:, 0], pts[:, 1], '-', color='tab:blue',
                    linewidth=1.0, alpha=0.35, zorder=3)

        incumbent = entry.get('incumbent_path')
        if incumbent is not None:
            pts = np.asarray(incumbent)
            ax.plot(pts[:, 0], pts[:, 1], '-o', color='red',
                    linewidth=2.0, markersize=3.0, alpha=0.95, zorder=4)

        ax.plot(q_xy[0], q_xy[1], 'go', markersize=8, zorder=6)
        ax.plot(goal_xy[0], goal_xy[1], 'r*', markersize=10, zorder=6)

        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        c_best_str = f"{c_best:.2f}" if np.isfinite(c_best) else "inf"
        ax.set_title(
            f"{title}  iter {entry['iteration']}  c_best={c_best_str}",
            fontsize=10,
        )
        ax.axis('off')
        fig.tight_layout(pad=0.3)

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(PILImage.open(buf).copy())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    durations = [1000 // fps] * len(frames)
    durations[-1] = 2000
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"Saved GIF → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_plans', type=int, default=100)
    parser.add_argument('--num_dreams', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--num_probes', type=int, default=1000)
    parser.add_argument('--time_budget', type=float, default=120)
    parser.add_argument('--map_file', type=str, default='intel.npy')
    parser.add_argument('--z', type=float, default=1.0)
    parser.add_argument('--radius', type=float, default=0.5)
    args = parser.parse_args()

    env, bounds, npy_path = build_environment(args.map_file, args.z, args.radius)

    vamp_cost_fn = CostVAMP.create(
        env,
        robot_name='sphere',
        n_jobs=1,
    )

    goals = GOAL_STATE[None, :]  # shape (1, 3)

    TENSOR_STATE = TensorState.create(
        dim=3,
        q=START_STATE,
        bounds=bounds,
        goals=goals,
        cost_fn=vamp_cost_fn,
        get_velocity=False,
        batch_size=args.num_plans,
        num_dreams=args.num_dreams,
        num_layers=args.num_layers,
        num_probes=args.num_probes,
    )

    print(f"Running with batch_size={args.num_plans}")
    print("Planning...")
    start_time = time.time()
    key = jax.random.PRNGKey(0)
    outputs, _ = ao_gtmp_plan(key, TENSOR_STATE, time_budget=args.time_budget)
    outputs.path.block_until_ready()
    print(f"Planning took {time.time() - start_time:.4f}s")

    iter_best_paths = outputs.timing.get('iter_best_paths', [])
    print(f"Recorded {len(iter_best_paths)} iterations for visualization")

    if len(iter_best_paths) > 0:
        out = get_gallery_path() / "benchmark_occupancy_ao_time_budget_vamp.gif"
        save_ao_progress_gif(
            iter_best_paths, npy_path, out,
            q_3d=np.asarray(START_STATE), goal_3d=np.asarray(GOAL_STATE),
            title="AO-GTMP + VAMP",
        )
    else:
        print("No iteration history recorded!")


if __name__ == "__main__":
    main()
