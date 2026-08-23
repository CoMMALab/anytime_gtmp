import argparse
import time
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

import jax
import jax.numpy as jnp
import sys
import os
import vamp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils.costs._cost_functions import CostVAMP

from planners.anytime_gtmp import TensorState, gtmp_plan_anytime
from planners.anytime_gtmp import *

from utils.costs._occupancy_map import build_environment
from pyroffi.costs import simplify_paths_batched

os.environ['JAX_PLATFORMS'] = 'cpu'
jax.config.update('jax_platform_name', 'cpu')


START_STATE = jnp.array([5.5, 4.9, 0.5])
GOAL_STATE = jnp.array([15.5, 16.2, 0.5])


def get_gallery_path():
    return Path(__file__).parent.parent / "gallery"


def expand_path_with_rrtc(waypoints, env, settings):
    """
    Re-run RRTConnect on each consecutive waypoint pair to recover the actual
    path through the map. Falls back to a straight line for any edge that
    cannot be re-solved.
    """
    rng = vamp.sphere.halton()
    detailed: list[np.ndarray] = []

    for i in range(len(waypoints) - 1):
        a = np.array(waypoints[i], dtype=np.float32)
        b = np.array(waypoints[i + 1], dtype=np.float32)
        result = vamp.sphere.rrtc(a, b, env, settings, rng)
        if result.solved:
            pts = result.path.numpy()
            if not detailed:
                detailed.append(pts)
            else:
                detailed.append(pts[1:])  # skip duplicate start point
        else:
            if not detailed:
                detailed.append(a[np.newaxis])
            detailed.append(b[np.newaxis])

    if not detailed:
        return np.array(waypoints)
    return np.concatenate(detailed, axis=0)


def save_paths_gif(paths, costs, npy_path, output_path, title="", fps=4, max_paths=None):
    """
    Animate collision-free paths as they are discovered. The running global-best
    (lowest cost so far) is drawn in red; all others in blue.

    paths: list of (N, 3) arrays already expanded via expand_path_with_rrtc,
           in discovery order.
    costs: list of scalar joint-space costs, aligned with `paths`.
    """
    map_data = np.load(npy_path)  # shape (H, W), 0=wall, 1=free

    if max_paths is not None:
        display_paths = paths[:max_paths]
        display_costs = costs[:max_paths]
    else:
        display_paths = paths
        display_costs = costs

    # Precompute the running-best index at each prefix length.
    best_idx_so_far = []
    running_best_cost = np.inf
    running_best_idx = 0
    for i, c in enumerate(display_costs):
        if c < running_best_cost:
            running_best_cost = c
            running_best_idx = i
        best_idx_so_far.append(running_best_idx)

    frames = []

    for n in range(1, len(display_paths) + 1):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
        # Transpose: axis 0 of .npy = x → columns in imshow
        ax.imshow(map_data.T, extent=[0, 20, 0, 20],
                  origin='lower', cmap='gray', alpha=0.65, aspect='auto')

        best_i = best_idx_so_far[n - 1]
        # Non-optimal paths first (blue, thinner) so the optimal draws on top.
        for i, path in enumerate(display_paths[:n]):
            if i == best_i:
                continue
            pts = np.asarray(path)
            ax.plot(pts[:, 0], pts[:, 1], '-',
                    color='tab:blue', linewidth=1.0, alpha=0.45)
        best_pts = np.asarray(display_paths[best_i])
        ax.plot(best_pts[:, 0], best_pts[:, 1], '-',
                color='red', linewidth=1.8, alpha=0.95, zorder=5)

        start_xy = np.asarray(display_paths[0][0, :2])
        goal_xy = np.asarray(display_paths[0][-1, :2])
        ax.plot(start_xy[0], start_xy[1], 'go', markersize=8, zorder=6, label='Start')
        ax.plot(goal_xy[0], goal_xy[1], 'r*', markersize=10, zorder=6, label='Goal')

        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_title(
            f"{title}  (paths: {n}, best cost: {display_costs[best_i]:.3f})",
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
    parser.add_argument('--rrtc_max_iterations', type=int, default=25)
    parser.add_argument('--map_file', type=str, default='intel.npy')
    parser.add_argument('--z', type=float, default=1.0)
    parser.add_argument('--radius', type=float, default=0.5)
    args = parser.parse_args()

    env, bounds, npy_path = build_environment(args.map_file, args.z, args.radius)

    print(f"Using RRTConnect validation with max_iterations={args.rrtc_max_iterations}")
    vamp_cost_fn = CostVAMP.create(
        env,
        robot_name='sphere',
        n_jobs=-2,
        validation_method='rrtc',
        validation_max_iterations=args.rrtc_max_iterations
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
    outputs, _ = gtmp_plan_anytime(
        key,
        TENSOR_STATE,
        min_layers=args.num_layers,
        max_layers=args.num_layers,
        min_dreams=args.num_dreams,
        max_dreams=args.num_dreams,
        time_budget=args.time_budget,
    )
    outputs.path.block_until_ready()
    planning_time = time.time() - start_time
    print(f"Planning took {planning_time:.4f}s")

    # Best collision-free path per iteration, in iteration order.
    iter_paths = outputs.timing.get('iter_best_paths', [])
    collisions = jax.device_get(outputs.collision)
    retained_count = int(np.sum(~collisions))

    print(f"\nResults:")
    print(f"  Validation method: RRTConnect (max_iterations={args.rrtc_max_iterations})")
    print(f"  Iterations with a feasible path: {len(iter_paths)}")
    print(f"  Retained {retained_count} in the final batch (top by cost)")
    print(f"  Total planning time: {planning_time:.4f}s")

    if len(iter_paths) > 0:
        first = iter_paths[0]['path']
        print(f"\nFirst path has {len(first)} waypoints")
        print(f"  Start: {first[0]}")
        print(f"  End:   {first[-1]}")

        vis_iter_paths = iter_paths[:100]

        def _edge_validator(a, b):
            if len(a) == 0:
                return np.zeros((0,), dtype=bool)
            return np.asarray(vamp_cost_fn._batch_edge_validation(a, b), dtype=bool)

        print(f"Shortcutting {len(vis_iter_paths)} per-iteration best paths...")
        batched = np.stack([np.asarray(entry['path']) for entry in vis_iter_paths], axis=0)
        simplified_batched, _ = simplify_paths_batched(
            batched,
            np.ones(len(vis_iter_paths), dtype=bool),
            edge_validator=_edge_validator,
            num_shortcut_rounds=64,
            seed=0,
        )
        shortcutted = [simplified_batched[i] for i in range(len(vis_iter_paths))]

        vis_settings = vamp.RRTCSettings()
        vis_settings.max_iterations = 2000
        print(f"Expanding {len(shortcutted)} shortcutted paths via RRTConnect for visualization...")
        expanded = [expand_path_with_rrtc(p, env, vis_settings) for p in shortcutted]
        costs = [
            float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=-1)))
            for p in shortcutted
        ]

        out = get_gallery_path() / "benchmark_occupancy_time_budget_rrtc.gif"
        save_paths_gif(expanded, costs, npy_path, out, title="GTMP + RRTConnect (anytime)")
    else:
        print("No paths found!")


if __name__ == "__main__":
    main()
