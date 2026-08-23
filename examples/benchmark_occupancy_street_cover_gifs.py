import argparse
import csv
import pickle
import tempfile
import time
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Ellipse
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image as PILImage

import jax
import jax.numpy as jnp
import sys
import os
import vamp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils.costs._cost_functions import CostVAMP

from planners.anytime_gtmp import TensorState, ao_gtmp_plan, gtmp_plan_anytime
from planners.anytime_gtmp import *

os.environ['JAX_PLATFORMS'] = 'cpu'
jax.config.update('jax_platform_name', 'cpu')


WORLD_SIZE = 20.0
SAMPLE_CMAP = 'viridis'          # node coloring for the per-iteration gradient (no red, so the red incumbent path reads clearly)
INCUMBENT_COLOR = "#ff0000"      # magenta incumbent / best path (absent from turbo, so it stays visible)
COSTBOUND_COLOR = 'red'        # cost-bound ellipse; cased in black below so it reads on any background
PRIOR_PATH_COLOR = 'tab:blue'    # faint earlier paths
PRIOR_PATH_LW = 1.6              # non-incumbent path line width (thicker = easier to see)
MAX_SAMPLES_PER_ITER = 120       # subsample drawn graph nodes per iteration for clarity

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.6,
})


def get_gallery_path():
    return Path(__file__).parent.parent / "gallery"


def get_street_png_path():
    return Path(__file__).parent.parent / "data" / "street-png"


def build_environment_from_png(png_path, z=1.0, radius=0.3, world=WORLD_SIZE):
    """Set up a vamp sphere environment from a binary street PNG.

    Dark pixels in the display PNG become wall columns for vamp. Returns
    (env, bounds, img_array) — img_array is the original (un-inverted) PNG so the
    visualization can display it in display orientation.
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
    """Re-run RRTConnect on each consecutive waypoint pair to recover the actual
    path through the map. Falls back to a straight line for any edge that cannot
    be re-solved."""
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
        fill=False, edgecolor=COSTBOUND_COLOR, linewidth=2.6, linestyle='--',
        zorder=5,
    )
    # Dark casing so the light dashed ellipse stays legible over bright turbo
    # samples and white free space alike.
    ellipse.set_path_effects([
        path_effects.withStroke(linewidth=4.2, foreground='black'),
    ])
    ax.add_patch(ellipse)


def _subsample_samples(samples):
    """Cap the number of drawn graph nodes for a single iteration."""
    if samples is None:
        return None
    pts = np.asarray(samples)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None
    if pts.shape[0] > MAX_SAMPLES_PER_ITER:
        sel = np.linspace(0, pts.shape[0] - 1, MAX_SAMPLES_PER_ITER).astype(int)
        pts = pts[sel]
    return pts


def _occ_is_free(disp, world, p, thresh=128):
    """True if world point `p` (xy) falls on a light (free) pixel of the map.

    `disp` is the displayed occupancy array (np.fliplr(img_array)); dark pixels
    are obstacles, matching how _draw_base_map renders the map.
    """
    H, W = disp.shape
    c = int(p[0] / world * W)
    r = int(p[1] / world * H)
    if r < 0 or r >= H or c < 0 or c >= W:
        return False
    return disp[r, c] >= thresh


def _free_points(pts, img_array, world, thresh=128):
    """Keep only sample points that fall on light (free) map pixels, dropping any
    that land on a dark obstacle pixel."""
    if pts is None:
        return None
    pts = np.asarray(pts)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None
    disp = np.fliplr(np.asarray(img_array))
    H, W = disp.shape
    cols = np.clip((pts[:, 0] / world * W).astype(int), 0, W - 1)
    rows = np.clip((pts[:, 1] / world * H).astype(int), 0, H - 1)
    free = disp[rows, cols] >= thresh
    return pts[free]


def _segment_free(disp, world, a, b):
    """Collision-check the straight segment a->b against the occupancy grid."""
    H, W = disp.shape
    cell = world / max(W, H)
    n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / cell) + 1)
    for t in np.linspace(0.0, 1.0, n):
        if not _occ_is_free(disp, world, a + t * (b - a)):
            return False
    return True


def _random_shortcut(path, img_array, world, seed=0, iters=200):
    """Randomized shortcut smoothing: repeatedly replace a random sub-segment of
    the polyline with a straight (collision-free) chord, so the path visibly
    relaxes toward something more optimal. Endpoints are preserved."""
    if path is None:
        return None
    pts = [np.asarray(p, dtype=float) for p in np.asarray(path)[:, :2]]
    if len(pts) <= 2:
        return np.asarray(pts)
    disp = np.fliplr(np.asarray(img_array))
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        if len(pts) <= 2:
            break
        i = int(rng.integers(0, len(pts) - 2))
        j = int(rng.integers(i + 2, len(pts)))
        if _segment_free(disp, world, pts[i], pts[j]):
            del pts[i + 1:j]
    return np.asarray(pts)


def _draw_base_map(ax, img_array, world):
    """Render the street map in display orientation (dark pixels = obstacles)."""
    ax.imshow(
        np.fliplr(img_array),
        extent=[0, world, 0, world],
        origin='lower',
        cmap='gray',
        alpha=0.85,
        aspect='equal',
    )


def _finalize_axes(ax, world):
    ax.set_xlim(0, world)
    ax.set_ylim(0, world)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def _frame_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).copy()


def _save_frames(frames, output_path, fps):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    durations = [1000 // fps] * len(frames)
    durations[-1] = 2500
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"Saved GIF → {output_path}")


def build_render_context(
    entries, method, img_array, q_3d, goal_3d, env, vis_settings,
    method_label, max_frames=40,
):
    """Pre-compute everything needed to render one planner's progress frames.

    Runs the expensive RRTC path expansions exactly once and packs the result
    into a plain (pickle-friendly) dict so the same data can drive the GIF, the
    static banner, and an on-disk dump for post-processing. Returns None when no
    iterations were recorded.

    method: 'ao' or 'anytime' — selects which per-iteration keys to read and how
    the incumbent / best path is tracked.
    """
    entries = entries[:max_frames]
    if not entries:
        print(f"No iterations recorded for {method_label}; skipping outputs.")
        return None

    q_xy = np.asarray(q_3d[:2], dtype=float)
    goal_xy = np.asarray(goal_3d[:2], dtype=float)
    c_min = float(np.linalg.norm(np.asarray(goal_3d, dtype=float) - np.asarray(q_3d, dtype=float)))

    # RRTC-expand each iteration's path once, cached by identity.
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

    def best_path_of(entry):
        if method == 'ao':
            return entry.get('incumbent_path')
        return entry.get('path')

    def iter_path_of(entry):
        if method == 'ao':
            return entry.get('iter_best_path')
        return entry.get('path')

    def cost_of(entry):
        if method == 'ao':
            return float(entry.get('c_best', np.inf))
        return float(entry.get('cost', np.inf))

    # Per-iteration expanded paths (for the incumbent and the faint priors), the
    # subsampled sample snapshots, and the raw dream points kept for post-tuning.
    best_expanded = [expand(best_path_of(e)) for e in entries]
    iter_expanded = [expand(iter_path_of(e)) for e in entries]
    samples_by_iter = [_subsample_samples(e.get('dream_points')) for e in entries]
    dream_points_raw = [
        None if e.get('dream_points') is None else np.asarray(e.get('dream_points'))
        for e in entries
    ]
    costs = [cost_of(e) for e in entries]
    iterations = [int(e.get('iteration', i + 1)) for i, e in enumerate(entries)]

    # Running best cost index (anytime tracks it manually; AO already reports
    # incumbent c_best, but the same logic is harmless there).
    running_best_idx = []
    best_cost = np.inf
    best_i = 0
    for i, c in enumerate(costs):
        if np.isfinite(c) and c < best_cost:
            best_cost = c
            best_i = i
        running_best_idx.append(best_i)

    n_iters = len(entries)
    return {
        'method': method,
        'method_label': method_label,
        'img_array': np.asarray(img_array),
        'q_xy': q_xy,
        'goal_xy': goal_xy,
        'c_min': c_min,
        'best_expanded': best_expanded,
        'iter_expanded': iter_expanded,
        'samples_by_iter': samples_by_iter,
        'dream_points_raw': dream_points_raw,
        'costs': costs,
        'iterations': iterations,
        'running_best_idx': running_best_idx,
        'n_iters': n_iters,
        'cmap_name': SAMPLE_CMAP,
        'norm_vmin': 1.0,
        'norm_vmax': float(max(n_iters, 2)),
    }


def _render_frame(ax, ctx, i, world, simplify_incumbent=False,
                  mask_collisions=False):
    """Draw a single progress frame (map + samples + paths + markers) onto `ax`.

    Returns (cmap, norm) so the caller can attach a colorbar. Captions, legend,
    and colorbars are left to the caller so the GIF and banner can style them
    differently.
    """
    _draw_base_map(ax, ctx['img_array'], world)

    cmap = plt.get_cmap(ctx['cmap_name'])
    norm = Normalize(vmin=ctx['norm_vmin'], vmax=ctx['norm_vmax'])

    # Accumulated GTMP samples, colored by the iteration they belong to. Older
    # samples stay slightly translucent so the newest layer reads clearly.
    samples_by_iter = ctx['samples_by_iter']
    for j in range(i + 1):
        pts = samples_by_iter[j]
        if pts is None:
            continue
        if mask_collisions:  # drop samples that land inside obstacles
            pts = _free_points(pts, ctx['img_array'], world)
            if pts is None or pts.shape[0] == 0:
                continue
        color = cmap(norm(j + 1))
        alpha = 0.55 + 0.40 * (j + 1) / (i + 1)
        ax.scatter(pts[:, 0], pts[:, 1], s=22, color=color, alpha=alpha,
                   edgecolors='black', linewidths=0.25, zorder=2)

    q_xy = ctx['q_xy']
    goal_xy = ctx['goal_xy']
    method = ctx['method']

    # Earlier (non-incumbent) paths, faint but drawn thick for legibility.
    if method == 'ao':
        for j in range(i + 1):
            prior_expanded = ctx['iter_expanded'][j]
            if prior_expanded is not None:
                ax.plot(prior_expanded[:, 0], prior_expanded[:, 1], '-',
                        color=PRIOR_PATH_COLOR, linewidth=PRIOR_PATH_LW, alpha=0.30, zorder=3)
        current_best = ctx['best_expanded'][i]
        # Cost-bound ellipse is only meaningful once a feasible path exists.
        if current_best is not None:
            _add_hyperellipsoid(ax, q_xy, goal_xy, ctx['costs'][i], ctx['c_min'])
    else:
        bi = ctx['running_best_idx'][i]
        for j in range(i + 1):
            if j == bi:
                continue
            pj = ctx['best_expanded'][j]
            if pj is not None:
                ax.plot(pj[:, 0], pj[:, 1], '-',
                        color=PRIOR_PATH_COLOR, linewidth=PRIOR_PATH_LW, alpha=0.30, zorder=3)
        current_best = ctx['best_expanded'][bi]

    if current_best is not None:
        if simplify_incumbent:
            # Visually relax the final path so it reads as more optimal than the
            # raw RRTC expansion (randomized, collision-aware shortcut smoothing).
            current_best = _random_shortcut(current_best, ctx['img_array'], world)
        ax.plot(current_best[:, 0], current_best[:, 1], '-',
                color=INCUMBENT_COLOR, linewidth=2.4, alpha=0.95, zorder=4,
                label='Incumbent path')

    ax.plot(q_xy[0], q_xy[1], 'o', color='#2ca02c', markersize=10,
            markeredgecolor='white', markeredgewidth=1.2, zorder=6, label='Start')
    ax.plot(goal_xy[0], goal_xy[1], '*', color=INCUMBENT_COLOR, markersize=18,
            markeredgecolor='white', markeredgewidth=1.0, zorder=6, label='Goal')

    _finalize_axes(ax, world)
    return cmap, norm


def build_method_gif(ctx, output_path, world=WORLD_SIZE, fps=4):
    """Render a progress GIF for one planner from a render context."""
    frames = []
    for i in range(ctx['n_iters']):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
        cmap, norm = _render_frame(ax, ctx, i, world)

        ax.set_title(ctx['method_label'], fontsize=16, fontweight='bold', pad=10)
        ax.text(
            0.02, 0.975,
            f"iteration {ctx['iterations'][i]}",
            transform=ax.transAxes, fontsize=12, va='top', ha='left',
            color='black',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='0.6', alpha=0.85),
            zorder=8,
        )
        ax.legend(loc='lower right', fontsize=10, framealpha=0.85, edgecolor='0.6')

        # Iteration-gradient colorbar for the samples.
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('GTMP sample iteration', fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        fig.tight_layout(pad=0.4)
        frames.append(_frame_to_image(fig))

    _save_frames(frames, output_path, fps)


def build_method_banner(ctx, output_path, world=WORLD_SIZE, n_frames=4):
    """Render a static `n_frames`-panel grid sampled evenly across the run."""
    n = ctx['n_iters']
    if n == 0:
        return
    k = min(n_frames, n)
    idxs = sorted(set(np.linspace(0, n - 1, k).astype(int).tolist()))

    ncols = int(np.ceil(np.sqrt(len(idxs))))
    nrows = int(np.ceil(len(idxs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.9 * nrows),
                             dpi=200, squeeze=False, layout='constrained')
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.06,
                                    wspace=0.03, hspace=0.06)
    flat_axes = axes.ravel()

    cmap = norm = None
    last_ax = flat_axes[0]
    for ax, i in zip(flat_axes, idxs):
        cmap, norm = _render_frame(ax, ctx, i, world,
                                   simplify_incumbent=(i == idxs[-1]),
                                   mask_collisions=True)
        ax.set_title(f"iteration {ctx['iterations'][i]}",
                     fontsize=13, pad=6)
        last_ax = ax
    for ax in flat_axes[len(idxs):]:  # hide any unused cells
        ax.axis('off')
    last_ax.legend(loc='lower right', fontsize=9, framealpha=0.9, edgecolor='0.6')

    # Centered horizontal colorbar under the grid keeps the layout symmetric.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                        fraction=0.045, pad=0.02, aspect=45, shrink=0.55)
    cbar.set_label('GTMP sample iteration', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(ctx['method_label'], fontsize=18, fontweight='bold')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white',
                pad_inches=0.12)
    plt.close(fig)
    print(f"Saved banner → {output_path}")


def save_render_data(ctx, output_path):
    """Pickle the render context so frames/banner can be re-tuned in post."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(ctx, f)
    print(f"Saved render data → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_plans', type=int, default=100)
    parser.add_argument('--num_dreams', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--num_probes', type=int, default=1000)
    parser.add_argument('--time_budget', type=float, default=60)
    parser.add_argument('--rrtc_max_iterations', type=int, default=10)
    parser.add_argument('--png_name', type=str, default='Berlin_0_1024.png',
                        help='PNG filename under data/street-png/ (MovingAI map).')
    parser.add_argument('--pair_index', type=int, default=0,
                        help='Which start/goal pair from start_goals.csv to use (0-indexed).')
    parser.add_argument('--start_goal_csv', type=str,
                        default=str(get_street_png_path() / 'start_goals.csv'))
    parser.add_argument('--radius', type=float, default=0.05)
    parser.add_argument('--max_frames', type=int, default=40)
    parser.add_argument('--banner_frames', type=int, default=4,
                        help='Number of evenly-spaced panels in the static banner PNG.')
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--methods', type=str, default='ao,anytime',
                        help="Comma list of planners to render: 'ao', 'anytime'.")
    args = parser.parse_args()

    png_path = get_street_png_path() / args.png_name
    if not png_path.exists():
        raise FileNotFoundError(f"PNG not found: {png_path}")

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
        validation_max_iterations=args.rrtc_max_iterations,
    )

    goals = goal_state[None, :]  # shape (1, 3)

    def make_state():
        return TensorState.create(
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

    vis_settings = vamp.RRTCSettings()
    vis_settings.max_iterations = 2000
    png_stem = Path(args.png_name).stem
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    def emit_outputs(ctx, stem):
        """Write the GIF, the static banner, and the picked render data."""
        gallery = get_gallery_path()
        build_method_gif(ctx, gallery / f"{stem}.gif", world=WORLD_SIZE, fps=args.fps)
        build_method_banner(ctx, gallery / f"{stem}_banner.png",
                            world=WORLD_SIZE, n_frames=args.banner_frames)
        save_render_data(ctx, gallery / f"{stem}_data.pkl")

    if 'ao' in methods:
        print("\n=== Running AO-GTMP (ao_gtmp_plan) ===")
        t0 = time.time()
        outputs, _ = ao_gtmp_plan(
            jax.random.PRNGKey(0), make_state(),
            time_budget=args.time_budget, record_samples=True,
        )
        outputs.path.block_until_ready()
        print(f"AO-GTMP planning took {time.time() - t0:.2f}s")
        entries = outputs.timing.get('iter_best_paths', [])
        print(f"Recorded {len(entries)} AO iterations")
        ctx = build_render_context(
            entries, 'ao', img_array,
            q_3d=start_np, goal_3d=goal_np, env=env, vis_settings=vis_settings,
            method_label='AO-GTMP', max_frames=args.max_frames,
        )
        if ctx is not None:
            emit_outputs(ctx, f"cover_ao_gtmp_{png_stem}_pair{args.pair_index}")

    if 'anytime' in methods:
        print("\n=== Running anytime-GTMP (gtmp_plan_anytime) ===")
        t0 = time.time()
        outputs, _ = gtmp_plan_anytime(
            jax.random.PRNGKey(0), make_state(),
            min_layers=args.num_layers, max_layers=args.num_layers,
            min_dreams=args.num_dreams, max_dreams=args.num_dreams,
            time_budget=args.time_budget, record_samples=True,
        )
        outputs.path.block_until_ready()
        print(f"anytime-GTMP planning took {time.time() - t0:.2f}s")
        entries = outputs.timing.get('iter_best_paths', [])
        print(f"Recorded {len(entries)} anytime iterations")
        ctx = build_render_context(
            entries, 'anytime', img_array,
            q_3d=start_np, goal_3d=goal_np, env=env, vis_settings=vis_settings,
            method_label='Anytime GTMP', max_frames=args.max_frames,
        )
        if ctx is not None:
            emit_outputs(ctx, f"cover_anytime_gtmp_{png_stem}_pair{args.pair_index}")


if __name__ == "__main__":
    main()
