import argparse
import csv
import os
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import random, vmap

import vamp
from planners.anytime_gtmp import (
    TensorState,
    sample_dream_points,
    gtmp_plan_externally_computed_costs,
)


WORLD_SIZE = 20.0


def get_gallery_path():
    return Path(__file__).parent.parent / "gallery"


def get_results_path():
    return Path(__file__).parent.parent / "results" / "aorrtc_local_connector"


def get_street_png_path():
    return Path(__file__).parent.parent / "data" / "street-png"




STATS_CSV_HEADER = [
    'budget',
    'global_cost',           # cost of the best GTMP path through the graph (inf if infeasible)
    'avg_local_edge_cost',   # mean AORRTC cost over collision-free edges (Cs + Ch + Cl)
    'num_valid_edges',       # count of finite-cost edges in the graph
    'num_total_edges',       # total candidate edges in the graph
    'aorrtc_seconds',
]


def edge_cost_summary(Cs, Ch, Cl):
    """Return (avg_finite_cost, num_valid, num_total) over all edges in the graph."""
    all_costs = np.concatenate([
        np.asarray(Cs).ravel(),
        np.asarray(Ch).ravel(),
        np.asarray(Cl).ravel(),
    ])
    finite = all_costs[np.isfinite(all_costs)]
    avg = float(finite.mean()) if finite.size > 0 else float('nan')
    return avg, int(finite.size), int(all_costs.size)


def build_environment_from_png(png_path, z=1.0, radius=0.3, world=WORLD_SIZE):
    """Set up a vamp sphere environment from a binary street PNG.

    Convention (matches examples/heightfield_coordinate_picker.py):
        displayed PNG  ⇒  black = collision, white = free.

    vamp.png_to_heightfield natively lifts BRIGHT pixels into wall columns, so
    we invert the PNG before passing it in. After inversion, originally-dark
    pixels (black in the visual display) become tall obstacles for vamp."""
    img = PILImage.open(png_path).convert('L')
    W, H = img.size
    arr = np.array(img)
    inverted = 255 - arr
    tmp_png = tempfile.mktemp(suffix='.png')
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
    img_array = np.array(img)
    return env, bounds, img_array


def load_pairs_from_csv(csv_path, png_name):
    """Return a list of (start, goal) tuples for `png_name` from the picker CSV.
    Each value is a length-3 np.float32 array (x, y, z)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    pairs = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get('png') != png_name:
                continue
            try:
                start = np.array([float(row['start_x']), float(row['start_y']),
                                  float(row['start_z'])], dtype=np.float32)
                goal = np.array([float(row['goal_x']), float(row['goal_y']),
                                 float(row['goal_z'])], dtype=np.float32)
            except (KeyError, ValueError):
                continue
            pairs.append((start, goal))
    return pairs


def aorrtc_all_edges(env, q, dream_points, goals, budget, optimize=True):
    """Run ``vamp.sphere.aorrtc_batch`` over every edge of the fixed graph.

    The edge layout matches CostVAMP._full_validate so the resulting cost
    matrices line up directly with the shapes expected by gtmp_plan-style
    value iteration:
        Cs (batch, num_dreams)
        Ch (batch, num_layers-1, num_dreams, num_dreams)
        Cl (batch, num_dreams, num_goals)
        Cg (num_goals,)

    Returns (Cs, Ch, Cl, Cg, edge_paths) where ``edge_paths`` is a dict
    with the three same-shape arrays of object dtype, each entry holding the
    dense waypoint array for the corresponding edge (or None when the AORRTC
    call failed). Infeasible edges have cost = ``inf``.
    """
    q = np.asarray(q, dtype=np.float32)
    dream_points = np.ascontiguousarray(np.asarray(dream_points, dtype=np.float32))
    goals = np.ascontiguousarray(np.asarray(goals, dtype=np.float32))
    batch_size, num_layers, num_dreams, dim = dream_points.shape
    num_goals = goals.shape[0]

    settings = vamp.AORRTCSettings()
    settings.max_iterations = budget
    settings.max_internal_iterations = budget
    settings.max_samples = budget
    settings.optimize = optimize

    pairs_a_blocks = []
    pairs_b_blocks = []

    # 1. start -> layer 0:  (batch_size * num_dreams) edges, in (b, n) order.
    pairs_a_blocks.append(np.tile(q, (batch_size * num_dreams, 1)))
    pairs_b_blocks.append(np.ascontiguousarray(dream_points[:, 0].reshape(-1, dim)))

    # 2. layer l -> layer l+1: (batch * (L-1) * N * N) edges, in (b, l, n_src, n_tgt).
    if num_layers > 1:
        dl = dream_points[:, :-1]  # (B, L-1, N, D)
        dl1 = dream_points[:, 1:]
        A = np.repeat(dl[:, :, :, np.newaxis, :], num_dreams, axis=3)
        B = np.tile(dl1[:, :, np.newaxis, :, :], (1, 1, num_dreams, 1, 1))
        pairs_a_blocks.append(np.ascontiguousarray(A.reshape(-1, dim)))
        pairs_b_blocks.append(np.ascontiguousarray(B.reshape(-1, dim)))

    # 3. last layer -> goals: (batch * N * num_goals) edges, in (b, n, g) order.
    last = dream_points[:, -1, :]  # (B, N, D)
    last_NG = np.tile(last[:, :, np.newaxis, :], (1, 1, num_goals, 1))
    goals_NG = np.tile(goals[np.newaxis, np.newaxis, :, :], (batch_size, num_dreams, 1, 1))
    pairs_a_blocks.append(np.ascontiguousarray(last_NG.reshape(-1, dim)))
    pairs_b_blocks.append(np.ascontiguousarray(goals_NG.reshape(-1, dim)))

    all_a = np.concatenate(pairs_a_blocks)
    all_b = np.concatenate(pairs_b_blocks)

    results = vamp.sphere.aorrtc_batch(all_a, all_b, env, settings)

    costs = np.fromiter(
        (float(r.path.cost()) if r.solved else float('inf') for r in results),
        dtype=np.float32, count=len(results),
    )
    paths = [r.path.numpy() if r.solved else None for r in results]

    Cs_size = batch_size * num_dreams
    Ch_size = batch_size * max(0, num_layers - 1) * num_dreams * num_dreams
    Cl_size = batch_size * num_dreams * num_goals

    offset = 0
    Cs = costs[offset:offset + Cs_size].reshape(batch_size, num_dreams)
    Cs_paths = np.empty((batch_size, num_dreams), dtype=object)
    for k in range(Cs_size):
        Cs_paths.flat[k] = paths[offset + k]
    offset += Cs_size

    if num_layers > 1:
        Ch = costs[offset:offset + Ch_size].reshape(
            batch_size, num_layers - 1, num_dreams, num_dreams,
        )
        Ch_paths = np.empty((batch_size, num_layers - 1, num_dreams, num_dreams), dtype=object)
        for k in range(Ch_size):
            Ch_paths.flat[k] = paths[offset + k]
        offset += Ch_size
    else:
        Ch = np.zeros((batch_size, 0, num_dreams, num_dreams), dtype=np.float32)
        Ch_paths = np.empty((batch_size, 0, num_dreams, num_dreams), dtype=object)

    Cl = costs[offset:offset + Cl_size].reshape(batch_size, num_dreams, num_goals)
    Cl_paths = np.empty((batch_size, num_dreams, num_goals), dtype=object)
    for k in range(Cl_size):
        Cl_paths.flat[k] = paths[offset + k]

    # By convention (matches the rest of the GTMP code), Cg is a flat vector
    # of -1's representing "free" goal action costs.
    Cg = -np.ones(num_goals, dtype=np.float32)

    edge_paths = {'start_to_layer': Cs_paths, 'layer_to_layer': Ch_paths, 'layer_to_goal': Cl_paths}
    return Cs, Ch, Cl, Cg, edge_paths


def stitched_path_for_plan(
    dream_points, goals, start, edge_paths, batch_idx,
    mid_idx_seq, goal_idx, num_layers, num_dreams, num_goals,
):
    """Concatenate the AORRTC dense paths for the (batch_idx)-th plan's chosen
    waypoint sequence into a single dense polyline.
    Returns None if any chosen edge has no path (AORRTC failed)."""
    segments = []

    # start -> dream_points[batch_idx, 0, mid_idx_seq[0]]
    s2l_path = edge_paths['start_to_layer'][batch_idx, mid_idx_seq[0]]
    if s2l_path is None:
        return None
    segments.append(s2l_path)

    # layer-to-layer edges
    for l in range(num_layers - 1):
        ptr = edge_paths['layer_to_layer'][batch_idx, l, mid_idx_seq[l], mid_idx_seq[l + 1]]
        if ptr is None:
            return None
        segments.append(ptr[1:])  # drop duplicate vertex

    # last layer -> selected goal
    g_path = edge_paths['layer_to_goal'][batch_idx, mid_idx_seq[-1], int(goal_idx)]
    if g_path is None:
        return None
    segments.append(g_path[1:])

    return np.concatenate(segments, axis=0)


def _all_edge_paths_for_batch(edge_paths, batch_idx):
    """Yield every dense AORRTC edge path for the given batch index, regardless
    of whether the planner ultimately selected it. Failed edges (None) are
    skipped. Used to paint the faint background of "all edges being optimized"."""
    s2l = edge_paths['start_to_layer'][batch_idx]   # (N,)
    l2l = edge_paths['layer_to_layer'][batch_idx]   # (L-1, N, N)
    l2g = edge_paths['layer_to_goal'][batch_idx]    # (N, G)
    for path in s2l.ravel():
        if path is not None:
            yield path
    for path in l2l.ravel():
        if path is not None:
            yield path
    for path in l2g.ravel():
        if path is not None:
            yield path


def _frame_for_budget(
    img_array, dream_points_batch, edge_paths, waypoints, connector,
    cost, budget, start, goal, world, title,
):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    # Display the PNG as-is (only re-oriented to match vamp's frame via
    # fliplr + origin='lower'). build_environment_from_png() already inverts
    # before handing the PNG to vamp, so dark pixels in this display are
    # vamp's obstacles.
    ax.imshow(
        np.fliplr(img_array),
        extent=[0, world, 0, world],
        origin='lower',
        cmap='gray',
        alpha=0.85,
        aspect='equal',
    )

    # Background: every AORRTC-optimized edge in the fixed graph, drawn
    # faintly so the viewer can see *all* edges getting refined as the budget
    # grows. We draw the chosen path on top of this layer.
    if edge_paths is not None:
        for path in _all_edge_paths_for_batch(edge_paths, batch_idx=0):
            ax.plot(
                path[:, 0], path[:, 1],
                '-', color='steelblue', linewidth=0.6, alpha=0.18, zorder=2,
            )

    # All graph nodes (dream points), per layer color for readability.
    if dream_points_batch is not None:
        num_layers = dream_points_batch.shape[0]
        cmap = plt.get_cmap('viridis')
        for li in range(num_layers):
            layer = dream_points_batch[li]  # (N, dim)
            ax.scatter(
                layer[:, 0], layer[:, 1],
                s=14, color=cmap(li / max(1, num_layers - 1)),
                alpha=0.85, edgecolor='black', linewidth=0.3, zorder=3,
                label=f'layer {li}' if li in (0, num_layers - 1) else None,
            )

    # Highlight the chosen path's intermediate waypoints.
    if len(waypoints) > 2:
        ax.plot(
            waypoints[1:-1, 0], waypoints[1:-1, 1],
            'o', color='orange', markersize=9, markeredgecolor='black',
            zorder=5, label='chosen waypoints',
        )

    # The chosen AORRTC-stitched dense connector, on top of everything else.
    if connector is not None and len(connector) > 1:
        ax.plot(
            connector[:, 0], connector[:, 1],
            '-', color='red', linewidth=2.6, alpha=0.95, zorder=6,
            label='best path',
        )

    ax.plot(start[0], start[1], 'o', color='lime', markersize=13,
            markeredgecolor='black', zorder=7, label='start')
    ax.plot(goal[0], goal[1], '*', color='gold', markersize=20,
            markeredgecolor='black', zorder=7, label='goal')

    ax.set_xlim(0, world)
    ax.set_ylim(0, world)
    cost_str = f"{cost:.2f}" if np.isfinite(cost) else "inf (infeasible)"
    ax.set_title(
        f"{title}\nAORRTC max_iterations = {budget}   GTMP path cost = {cost_str}",
        fontsize=11,
    )
    ax.legend(loc='lower right', fontsize=8, framealpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0.4)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).copy()


def _decode_mid_indices(state, dream_points_jax, Cs, Ch, Cl, Cg, Vh, gamma):
    """Re-derive the chosen (mid_idx_seq, goal_idx) for every plan in the batch
    using the same recurrence as get_optimal_path but in numpy on host."""
    Cs = np.asarray(Cs); Ch = np.asarray(Ch)
    Cl = np.asarray(Cl); Cg = np.asarray(Cg); Vh = np.asarray(Vh)
    batch_size, num_layers, num_dreams = Vh.shape[0], state.num_layers, state.num_dreams

    current = np.argmin(Cs + gamma * Vh[:, 0], axis=-1)  # (B,)
    mid_seq = [current.copy()]
    for i in range(1, num_layers):
        # Ch shape (B, L-1, N_src, N_tgt). For each batch b, pick Ch[b, i-1, current[b], :].
        ch_slice = np.take_along_axis(
            Ch[:, i - 1], current[:, None, None].repeat(num_dreams, axis=-1), axis=1,
        ).squeeze(axis=1)
        current = np.argmin(ch_slice + gamma * Vh[:, i], axis=-1)
        mid_seq.append(current.copy())
    mid_seq = np.stack(mid_seq, axis=1)  # (B, num_layers)

    cl_slice = np.take_along_axis(Cl, current[:, None, None].repeat(Cl.shape[-1], axis=-1), axis=1).squeeze(axis=1)
    goal_idx = np.argmin(cl_slice + gamma * Cg, axis=-1)  # (B,)
    return mid_seq, goal_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--png', default='Berlin_0_1024.png',
                        help='PNG filename under data/street-png/.')
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_dreams', type=int, default=100)
    parser.add_argument('--num_plans', type=int, default=1)
    parser.add_argument('--gif_budgets', type=int, nargs='+',
                        default=[5,10,25,50,100,250, 500, 1000])
    parser.add_argument('--radius', type=float, default=0.05)
    parser.add_argument('--z', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default=None)
    parser.add_argument('--start_goal_csv',
                        default=str(get_street_png_path() / 'start_goals.csv'),
                        help='CSV produced by examples/heightfield_coordinate_picker.py. '
                             'A row for --png is REQUIRED — there is no auto-search fallback.')
    parser.add_argument('--pair_index', type=int, default=0,
                        help='When the CSV has multiple pairs for the same PNG, pick this one.')
    args = parser.parse_args()

    # Accept --png as either (a) a bare filename inside data/street-png/, or
    # (b) any path that resolves on its own (absolute, or relative to cwd, or
    # already prefixed with data/street-png/). Callers wiring this script from
    # a shell loop typically pass the full relative path.
    raw = Path(args.png)
    if raw.exists():
        png_path = raw
    else:
        png_path = get_street_png_path() / raw.name
        if not png_path.exists():
            raise FileNotFoundError(
                f"PNG not found. Tried {raw} and {png_path}."
            )
    env, bounds, img_array = build_environment_from_png(
        png_path, z=1.0, radius=args.radius, world=WORLD_SIZE,
    )

    pairs = load_pairs_from_csv(args.start_goal_csv, png_path.name)
    if not pairs:
        raise RuntimeError(
            f"No start/goal pair for {args.png} in {args.start_goal_csv}. "
            f"Run examples/heightfield_coordinate_picker.py --png {args.png} first."
        )
    if args.pair_index >= len(pairs):
        raise RuntimeError(
            f"--pair_index={args.pair_index} but only {len(pairs)} pair(s) "
            f"in {args.start_goal_csv} for {args.png}"
        )
    start, goal = pairs[args.pair_index]
    if not (vamp.sphere.validate(start, env) and vamp.sphere.validate(goal, env)):
        raise RuntimeError(
            f"CSV pair {args.pair_index} for {args.png} is in collision under "
            f"radius={args.radius}, z={args.z}. Re-pick or adjust radius."
        )
    print(f"Start: {tuple(float(v) for v in start)}")
    print(f"Goal:  {tuple(float(v) for v in goal)}")

    START_STATE = jnp.asarray(start, dtype=jnp.float32)
    GOAL_STATE = jnp.asarray(goal, dtype=jnp.float32)
    goals_jax = GOAL_STATE[None, :]

    tensor_state = TensorState.create(
        dim=3,
        q=START_STATE,
        bounds=bounds,
        goals=goals_jax,
        cost_fn=None,
        batch_size=args.num_plans,
        num_layers=args.num_layers,
        num_dreams=args.num_dreams,
    )

    # Sample the fixed graph ONCE — same dream points across the budget sweep.
    rng = random.PRNGKey(args.seed)
    keys = random.split(rng, args.num_plans)
    dream_points = vmap(
        lambda k: sample_dream_points(k, tensor_state.bounds,
                                      (args.num_layers, args.num_dreams),
                                      dtype=jnp.float32)
    )(keys)
    dream_points.block_until_ready()
    dream_points_np = np.asarray(jax.device_get(dream_points))
    goals_np = np.asarray(jax.device_get(goals_jax))
    print(
        f"Fixed graph: batch={args.num_plans}, layers={args.num_layers}, "
        f"dreams={args.num_dreams}, total edges per budget = "
        f"{args.num_plans * (args.num_dreams + max(0, args.num_layers - 1) * args.num_dreams ** 2 + args.num_dreams)}"
    )

    title = f"{png_path.name}  fixed graph ({args.num_layers}×{args.num_dreams})  pair #{args.pair_index}"
    gamma_np = np.float32(1.0)

    stats_rows = []
    frames = []
    for B in args.gif_budgets:
        t0 = time.time()
        Cs_np, Ch_np, Cl_np, Cg_np, edge_paths = aorrtc_all_edges(
            env, np.asarray(start), dream_points_np, goals_np,
            budget=B, optimize=True,
        )
        t_aorrtc = time.time() - t0
        avg_edge_cost, num_valid_edges, num_total_edges = edge_cost_summary(Cs_np, Ch_np, Cl_np)

        # Run value iteration + path extraction on the AORRTC cost matrices.
        Cs_j = jnp.asarray(Cs_np); Ch_j = jnp.asarray(Ch_np)
        Cl_j = jnp.asarray(Cl_np); Cg_j = jnp.asarray(Cg_np)
        outputs = gtmp_plan_externally_computed_costs(
            tensor_state, dream_points, Cs_j, Ch_j, Cl_j, Cg_j,
        )
        outputs.path.block_until_ready()
        t_total = time.time() - t0

        collisions = np.asarray(jax.device_get(outputs.collision))
        feasible_idx = np.where(~collisions)[0]
        if feasible_idx.size == 0:
            print(
                f"  budget={B:<5d}  no feasible plan in graph; "
                f"valid_edges={num_valid_edges}/{num_total_edges}, "
                f"avg_local={avg_edge_cost:.3f}  "
                f"(aorrtc {t_aorrtc:.2f}s, total {t_total:.2f}s)"
            )
            stats_rows.append({
                'budget': B,
                'global_cost': float('inf'),
                'avg_local_edge_cost': avg_edge_cost,
                'num_valid_edges': num_valid_edges,
                'num_total_edges': num_total_edges,
                'aorrtc_seconds': t_aorrtc,
            })
            frames.append(_frame_for_budget(
                img_array, dream_points_batch=dream_points_np[0],
                edge_paths=edge_paths,
                waypoints=np.stack([np.asarray(start), np.asarray(goal)]),
                connector=None, cost=float('inf'),
                budget=B, start=start, goal=goal, world=WORLD_SIZE, title=title,
            ))
            continue

        Vh = np.asarray(jax.device_get(outputs.V))
        Vs = np.asarray(jax.device_get(outputs.Vs))
        mid_seq_batch, goal_idx_batch = _decode_mid_indices(
            tensor_state, dream_points, Cs_np, Ch_np, Cl_np, Cg_np, Vh, gamma_np,
        )

        # Of the feasible plans, pick the one with smallest Vs (the joint
        # graph cost — equals the AORRTC-stitched cost up to the trailing
        # gamma * Cg = -1 term, which is constant across plans).
        best_b = int(feasible_idx[np.argmin(Vs[feasible_idx])])
        mid_seq = mid_seq_batch[best_b]
        goal_idx = int(goal_idx_batch[best_b])

        # Rebuild the chosen waypoint polyline (just the graph vertices).
        chosen_waypoints = np.concatenate([
            np.asarray(start, dtype=np.float32)[None, :],
            dream_points_np[best_b, np.arange(args.num_layers), mid_seq, :],
            np.asarray(goal, dtype=np.float32)[None, :],
        ], axis=0)

        connector = stitched_path_for_plan(
            dream_points_np, goals_np, start, edge_paths,
            batch_idx=best_b, mid_idx_seq=mid_seq, goal_idx=goal_idx,
            num_layers=args.num_layers, num_dreams=args.num_dreams,
            num_goals=goals_np.shape[0],
        )
        if connector is not None:
            connector_cost = float(np.sum(np.linalg.norm(np.diff(connector, axis=0), axis=-1)))
        else:
            connector_cost = float('inf')

        # GTMP's Vs already accounts for Cs + Ch + Cl + gamma*Cg. With gamma=1
        # and Cg = -1 (single-goal default), Vs = sum-of-AORRTC-edge-costs - 1.
        graph_cost = float(Vs[best_b]) - float(gamma_np) * float(Cg_np[goal_idx])
        print(
            f"  budget={B:<5d}  GTMP plan #{best_b}  graph cost={graph_cost:.3f}  "
            f"connector polyline cost={connector_cost:.3f}  "
            f"valid_edges={num_valid_edges}/{num_total_edges}  "
            f"avg_local={avg_edge_cost:.3f}  "
            f"(aorrtc {t_aorrtc:.2f}s, total {t_total:.2f}s)"
        )
        stats_rows.append({
            'budget': B,
            'global_cost': graph_cost,
            'avg_local_edge_cost': avg_edge_cost,
            'num_valid_edges': num_valid_edges,
            'num_total_edges': num_total_edges,
            'aorrtc_seconds': t_aorrtc,
        })

        frames.append(_frame_for_budget(
            img_array, dream_points_batch=dream_points_np[best_b],
            edge_paths=edge_paths,
            waypoints=chosen_waypoints, connector=connector,
            cost=graph_cost, budget=B,
            start=start, goal=goal, world=WORLD_SIZE, title=title,
        ))

    # Default output goes under results/aorrtc_local_connector/. The GIF and
    # the per-budget stats CSV share a stem so they live side-by-side.
    problem_stem = f"{png_path.stem}__pair{args.pair_index}"
    if args.output:
        output_path = Path(args.output)
        stats_path = output_path.with_suffix('.csv')
    else:
        results_dir = get_results_path()
        output_path = results_dir / f"{problem_stem}.gif"
        stats_path = results_dir / f"{problem_stem}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    durations = [900] * len(frames)
    durations[-1] = 2400
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"Saved GIF → {output_path}")

    with stats_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=STATS_CSV_HEADER)
        w.writeheader()
        for row in stats_rows:
            w.writerow(row)
    print(f"Saved stats → {stats_path}")


if __name__ == "__main__":
    main()
