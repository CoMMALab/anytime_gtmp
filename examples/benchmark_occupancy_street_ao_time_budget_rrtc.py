import argparse
import csv
import tempfile
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
import vamp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils.costs._cost_functions import CostVAMP

from planners.anytime_gtmp import TensorState, ao_gtmp_plan
from planners.anytime_gtmp import *

os.environ['JAX_PLATFORMS'] = 'cpu'
jax.config.update('jax_platform_name', 'cpu')


WORLD_SIZE = 20.0


def get_gallery_path():
    return Path(__file__).parent.parent / "gallery"


def get_street_png_path():
    return Path(__file__).parent.parent / "data" / "street-png"


def build_environment_from_png(png_path, z=1.0, radius=0.3, world=WORLD_SIZE):
    """Set up a vamp sphere environment from a binary street PNG.

    Mirrors examples/benchmark_occupancy_street_aorrtc_local_connector.py:
    dark pixels in the display PNG become wall columns for vamp.
    Returns (env, bounds, img_array) — img_array is the original (un-inverted)
    PNG so the visualization can display it in display orientation.
    """
    img = PILImage.open(png_path).convert('L')
    W, H = img.size
    arr = np.array(img)
    inverted = 255 - arr
    tmp_fd, tmp_png = tempfile.mkstemp(suffix='.png')
    os.close(tmp_fd)
    PILImage.fromarray(inverted).save(tmp_png)

    env = vamp.Environment()
    hf = vamp.png_to_heightfield(
        tmp_png,
        (world / 2.0, world / 2.0, 0.0),
        (world / W, world / H, z),
    )
    env.add_heightfield(hf)

    vamp.sphere.set_lows([0.0, 0.0, 0.0])
    vamp.sphere.set_highs([world, world, z])
    vamp.sphere.set_radius(radius)

    bounds = jnp.array([[0.0, world], [0.0, world], [0.0, z]])
    return env, bounds, arr


def load_pair_from_csv(csv_path, png_name, pair_index=0):
    """Return the pair_index'th (start, goal) for `png_name` in start_goals.csv."""
    csv_path = Path(csv_path)
    pairs = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get('png') != png_name:
                continue
            start = np.array([float(row['start_x']), float(row['start_y']),
                              float(row['start_z'])], dtype=np.float32)
            goal = np.array([float(row['goal_x']), float(row['goal_y']),
                             float(row['goal_z'])], dtype=np.float32)
            pairs.append((start, goal))
    if not pairs:
        raise ValueError(f"No start/goal entry for {png_name} in {csv_path}")
    if pair_index < 0 or pair_index >= len(pairs):
        raise IndexError(
            f"pair_index {pair_index} out of range (0..{len(pairs) - 1}) for {png_name}"
        )
    return pairs[pair_index]


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
    iter_best_paths, img_array, output_path, q_3d, goal_3d, env, vis_settings,
    world=WORLD_SIZE, title="", fps=4, max_frames=40,
):
    """Animate AO-GTMP iterations: incumbent path (RRTC-expanded) + hyperellipsoid."""
    entries = iter_best_paths[:max_frames]

    q_xy = np.asarray(q_3d[:2], dtype=float)
    goal_xy = np.asarray(goal_3d[:2], dtype=float)
    c_min = float(np.linalg.norm(np.asarray(goal_3d, dtype=float) - np.asarray(q_3d, dtype=float)))

    # Cache expansions keyed by path identity to avoid redundant RRTC calls
    # when the incumbent is unchanged across iterations.
    expansion_cache: dict = {}

    def expand(path):
        if path is None:
            return None
        key = id(path)
        cached = expansion_cache.get(key)
        if cached is not None:
            return cached
        expanded = expand_path_with_rrtc(np.asarray(path), env, vis_settings)
        expansion_cache[key] = expanded
        return expanded

    # Pre-expand each iteration's best path once so blue overlays are cheap per-frame.
    for entry in entries:
        expand(entry.get('iter_best_path'))
        expand(entry.get('incumbent_path'))

    frames = []
    for i, entry in enumerate(entries):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
        # Display the street PNG in display orientation. build_environment_from_png
        # already inverts before handing the PNG to vamp, so dark pixels here
        # correspond to vamp's obstacles (matches the convention in
        # benchmark_occupancy_street_aorrtc_local_connector.py).
        ax.imshow(
            np.fliplr(img_array),
            extent=[0, world, 0, world],
            origin='lower',
            cmap='gray',
            alpha=0.85,
            aspect='equal',
        )

        c_best = float(entry['c_best'])
        _add_hyperellipsoid(ax, q_xy, goal_xy, c_best, c_min)

        for prior in entries[:i + 1]:
            prior_expanded = expand(prior.get('iter_best_path'))
            if prior_expanded is None:
                continue
            ax.plot(prior_expanded[:, 0], prior_expanded[:, 1], '-',
                    color='tab:blue', linewidth=1.0, alpha=0.35, zorder=3)

        incumbent_expanded = expand(entry.get('incumbent_path'))
        if incumbent_expanded is not None:
            ax.plot(incumbent_expanded[:, 0], incumbent_expanded[:, 1], '-',
                    color='red', linewidth=2.0, alpha=0.95, zorder=4)

        ax.plot(q_xy[0], q_xy[1], 'go', markersize=8, zorder=6)
        ax.plot(goal_xy[0], goal_xy[1], 'r*', markersize=10, zorder=6)

        ax.set_xlim(0, world)
        ax.set_ylim(0, world)
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
    parser.add_argument('--rrtc_max_iterations', type=int, default=10)
    parser.add_argument('--png_name', type=str, default='Berlin_0_1024.png',
                        help='PNG filename under data/street-png/.')
    parser.add_argument('--pair_index', type=int, default=0,
                        help='Which start/goal pair from start_goals.csv to use (0-indexed).')
    parser.add_argument('--start_goal_csv', type=str,
                        default=str(get_street_png_path() / 'start_goals.csv'))
    parser.add_argument('--radius', type=float, default=0.05)
    args = parser.parse_args()

    png_path = get_street_png_path() / args.png_name
    if not png_path.exists():
        raise FileNotFoundError(f"PNG not found: {png_path}")

    # Heightfield z hardcoded to 1.0 to match
    # benchmark_occupancy_street_aorrtc_local_connector.py.
    env, bounds, img_array = build_environment_from_png(
        png_path, z=1.0, radius=args.radius, world=WORLD_SIZE,
    )

    start_np, goal_np = load_pair_from_csv(args.start_goal_csv, args.png_name, args.pair_index)
    if not (vamp.sphere.validate(start_np, env) and vamp.sphere.validate(goal_np, env)):
        raise RuntimeError(
            f"Start or goal is in collision for {args.png_name} pair {args.pair_index}."
        )
    start_state = jnp.asarray(start_np, dtype=jnp.float32)
    goal_state = jnp.asarray(goal_np, dtype=jnp.float32)
    print(f"Using {args.png_name} pair {args.pair_index}: "
          f"start={start_np.tolist()} goal={goal_np.tolist()}")

    print(f"Using RRTConnect validation with max_iterations={args.rrtc_max_iterations}")
    vamp_cost_fn = CostVAMP.create(
        env,
        robot_name='sphere',
        n_jobs=-2,
        validation_method='rrtc',
        validation_max_iterations=args.rrtc_max_iterations
    )

    goals = goal_state[None, :]  # shape (1, 3)

    TENSOR_STATE = TensorState.create(
        dim=3,
        q=start_state,
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
    planning_time = time.time() - start_time
    print(f"Planning took {planning_time:.4f}s")

    iter_best_paths = outputs.timing.get('iter_best_paths', [])
    print(f"Recorded {len(iter_best_paths)} iterations for visualization")

    if len(iter_best_paths) > 0:
        vis_settings = vamp.RRTCSettings()
        vis_settings.max_iterations = 2000
        png_stem = Path(args.png_name).stem
        out = (
            get_gallery_path()
            / f"benchmark_occupancy_street_ao_time_budget_rrtc_{png_stem}_pair{args.pair_index}.gif"
        )
        save_ao_progress_gif(
            iter_best_paths, img_array, out,
            q_3d=start_np, goal_3d=goal_np,
            env=env, vis_settings=vis_settings,
            world=WORLD_SIZE,
            title=f"AO-GTMP + RRTConnect — {args.png_name}",
        )
    else:
        print("No iteration history recorded!")


if __name__ == "__main__":
    main()
