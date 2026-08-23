# Force JAX to use CPU only
import os
# Configure JAX to use CPU
os.environ['JAX_PLATFORMS'] = 'cpu'


# from utils.sampling.generate_configurations import load_panda_robot
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import vmap, random, jit, lax
from flax import struct
from typing import Tuple, Optional, Union, Any, List, Dict
from functools import partial
import pandas as pd
import time
from dataclasses import dataclass, field, replace
import numpy as np
from utils.analytics import entropy_path
from pyroffi.costs import simplify_paths_batched


@jax.tree_util.register_pytree_node_class
@dataclass
class TensorState:
    # JAX arrays (leaves)
    q: Optional[jnp.ndarray] = None
    bounds: Optional[jnp.ndarray] = None
    goals: Optional[jnp.ndarray] = None
    time_profile: Optional[jnp.ndarray] = None
    probes: Optional[jnp.ndarray] = None
    presampled_configs: Optional[jnp.ndarray] = None
    presampled_fk_results: Optional[jnp.ndarray] = None

    # Static fields
    dim: int = 2
    batch_size: Optional[int] = None   
    splines: Any = None
    num_dreams: int = 50
    num_layers: int = 5
    num_probes: int = 10
    cost_fn: Any = None
    cell_size: float = 1.0
    scale_objective: float = 1.0
    scale_occ: float = 1.0
    scale_dist: float = 1.0
    gamma: float = 0.99
    vi_finite: bool = True
    sampling_free_space: bool = False
    visualize_value: bool = False
    get_velocity: bool = False
    dtype: jnp.dtype = jnp.float32
    robot: Any = None

    def tree_flatten(self):
        leaves = (
            self.q,
            self.bounds,
            self.goals,
            self.time_profile,
            self.probes,
            self.presampled_configs,
            self.presampled_fk_results,
        )
        aux_data = {
            'dim': self.dim,
            'batch_size': self.batch_size,   
            'splines': self.splines,
            'num_dreams': self.num_dreams,
            'num_layers': self.num_layers,
            'num_probes': self.num_probes,
            'cost_fn': self.cost_fn,
            'cell_size': self.cell_size,
            'scale_objective': self.scale_objective,
            'scale_occ': self.scale_occ,
            'scale_dist': self.scale_dist,
            'gamma': self.gamma,
            'vi_finite': self.vi_finite,
            'sampling_free_space': self.sampling_free_space,
            'visualize_value': self.visualize_value,
            'get_velocity': self.get_velocity,
            'dtype': self.dtype,
            'robot': self.robot,
        }
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        return cls(
            q=leaves[0],
            bounds=leaves[1],
            goals=leaves[2],
            time_profile=leaves[3],
            probes=leaves[4],
            presampled_configs=leaves[5],
            presampled_fk_results=leaves[6],
            **aux_data
        )

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    @classmethod
    def create(
        cls,
        dim: int = 2,
        batch_size: Optional[int] = None,   
        q: jax.Array = None,
        bounds: jax.Array = None,
        goals: jax.Array = None,
        time_profile: jax.Array = None,
        splines: Any = None,
        cost_fn: Any = None,
        scale_objective: Optional[float] = 1.,
        scale_occ: Optional[float] = 1.,
        scale_dist: Optional[float] = 1.,
        gamma: float = 0.99,
        cell_size: float = 1.0,
        num_dreams: int = 50,
        num_layers: int = 5,
        num_probes: int = 10,
        vi_finite: bool = True,
        sampling_free_space: bool = False,
        visualize_value: bool = False,
        get_velocity: bool = False,
        dtype: Optional[jnp.dtype] = jnp.float32,
        robot: Any = None,
        **kwargs: Any,
    ) -> "TensorState":

        probes = jnp.linspace(0, 1, num_probes + 2)[:-1]

        if time_profile is None:
            time_profile = jnp.linspace(
                0, num_layers + 1, num_layers + 2, dtype=dtype
            )

        return cls(
            q=q,
            dim=q.shape[-1] if q is not None else dim,
            batch_size=batch_size,          # ✅ NEW
            cost_fn=cost_fn,
            bounds=bounds,
            goals=goals,
            time_profile=time_profile,
            splines=splines,
            num_dreams=num_dreams,
            num_layers=num_layers,
            num_probes=num_probes,
            probes=probes,
            scale_objective=scale_objective,
            scale_occ=scale_occ,
            scale_dist=scale_dist,
            cell_size=cell_size,
            gamma=gamma,
            vi_finite=vi_finite,
            sampling_free_space=sampling_free_space,
            visualize_value=visualize_value,
            get_velocity=get_velocity,
            dtype=dtype,
            robot=robot,
            **kwargs,
        )


    def update_num_layer(self, num_layers: int) -> "TensorState":
        time_profile = jnp.linspace(0, num_layers + 1, num_layers + 2, dtype=self.dtype)
        return self.replace(num_layers=num_layers, time_profile=time_profile)


@partial(jit, static_argnums=(2, 3))
def sample_dream_points(rng: jax.Array, bounds: jax.Array, num_dreams: Tuple[int] = (100,), dtype: jnp.dtype = jnp.float32) -> jax.Array:
    rng, sub_rng = random.split(rng)
    dim = bounds.shape[0]
    return random.uniform(sub_rng, num_dreams + (dim,), dtype, minval=bounds[:, 0], maxval=bounds[:, 1])

    
@partial(jit, static_argnums=6)
def value_iteration(Cs: jax.Array, Ch: jax.Array, Cl: jax.Array, Cg: jax.Array,
                    gamma: float = 0.9, eps: float = 1e-2, dtype: jnp.dtype = jnp.float32,
                    Vs_init: Optional[float] = None, Vh_init: Optional[jax.Array] = None) -> jax.Array:
    num_layer, num_dreams = Ch.shape[0] + 1, Ch.shape[1]
    Cs = jnp.asarray(Cs, dtype=dtype)
    Ch = jnp.asarray(Ch, dtype=dtype)
    Cl = jnp.asarray(Cl, dtype=dtype)
    Cg = jnp.asarray(Cg, dtype=dtype)
    
    # Use previous solution if provided, otherwise zero initialization
    Vs = jnp.asarray(Vs_init, dtype=dtype) if Vs_init is not None else jnp.asarray(0.0, dtype=dtype)
    Vh = jnp.asarray(Vh_init, dtype=dtype) if Vh_init is not None else jnp.zeros((num_layer, num_dreams), dtype=dtype)

    # Vs = 0.
    # Vh = jnp.zeros((num_layer, num_dreams), dtype=dtype)
    

    def optimal_bellman(V_tup: Tuple[float, float, jax.Array, jax.Array]) -> Tuple[float, float, jax.Array, jax.Array]:
        pVs, Vs, pVh, Vh = V_tup
        pVs, pVh = Vs, Vh
        Vh = Vh.at[-1].set(jnp.min(Cl + gamma * Cg, axis=-1))
        Vh = Vh.at[:num_layer - 1].set(jnp.min(Ch + gamma * Vh[1:num_layer, None, :], axis=-1))
        Vs = jnp.min(Cs + gamma * Vh[0])
        return pVs, Vs, pVh, Vh

    V_tup = optimal_bellman((Vs, Vs, Vh, Vh))
    tol = eps * (1 - gamma) / gamma
    _, Vs, _, Vh = lax.while_loop(lambda V: jnp.abs(V[0] - V[1]) > tol, optimal_bellman, V_tup)
    return Vs, Vh


@partial(jit, static_argnums=4)
def value_iteration_finite(Cs: jax.Array, Ch: jax.Array, Cl: jax.Array, Cg: jax.Array, dtype: jnp.dtype = jnp.float32,
                           Vs_init: Optional[jax.Array] = None, Vh_init: Optional[jax.Array] = None) -> jax.Array:
    # Always expects batched inputs:
    # Cs (batch_size, num_dreams), Ch (batch_size, num_layers-1, num_dreams, num_dreams)
    Cs = jnp.asarray(Cs, dtype=dtype)
    Ch = jnp.asarray(Ch, dtype=dtype)
    Cl = jnp.asarray(Cl, dtype=dtype)
    Cg = jnp.asarray(Cg, dtype=dtype)
    batch_size = Cs.shape[0]
    num_layer, num_dreams = Ch.shape[1] + 1, Ch.shape[2]
    Vs = jnp.asarray(Vs_init, dtype=dtype) if Vs_init is not None else jnp.zeros(batch_size, dtype=dtype)
    Vh = jnp.asarray(Vh_init, dtype=dtype) if Vh_init is not None else jnp.zeros((batch_size, num_layer, num_dreams), dtype=dtype)

    def optimal_bellman(i:int, V_tup: Tuple[jax.Array, jax.Array, jax.Array, jax.Array]) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        pVs, Vs, pVh, Vh = V_tup
        pVs, pVh = Vs, Vh
        Vh = Vh.at[:, -1].set(jnp.min(Cl + Cg, axis=-1))
        Vh = Vh.at[:, :num_layer - 1].set(jnp.min(Ch + Vh[:, 1:num_layer, None, :], axis=-1))
        Vs = jnp.min(Cs + Vh[:, 0], axis=-1)
        return pVs, Vs, pVh, Vh

    V_tup = optimal_bellman(0, (Vs, Vs, Vh, Vh))
    T = num_layer + 1
    _, Vs, _, Vh = lax.fori_loop(1, T, optimal_bellman, V_tup)
    return Vs, Vh


@jit
def get_optimal_path(Cs: jax.Array, Ch: jax.Array, Cl: jax.Array, Cg: jax.Array, Vh: jax.Array, gamma: float = 0.9) -> jax.Array:
    # Always expects batched inputs
    batch_size = Cs.shape[0]
    num_layer = Vh.shape[1]
    current = jnp.argmin(Cs + gamma * Vh[:, 0], axis=-1)
    path = [current]
    batch_idx = jnp.arange(batch_size)
    for i in range(1, num_layer):
        current = jnp.argmin(Ch[batch_idx, i - 1, current] + gamma * Vh[:, i], axis=-1)
        path.append(current)
    goal_id = jnp.argmin(Cl[batch_idx, current] + gamma * Cg, axis=-1)
    return jnp.stack(path, axis=1), goal_id


@struct.dataclass
class TensorOutput():

    path: jax.Array = None
    path_vel: jax.Array = None
    goal_idx: int = None
    collision: bool = False
    dream_points: jax.Array = None
    # TODO: Revisit this to in case any type affects output
    splines: Any = None
    V: jax.Array = None
    Vs: jax.Array = None  # Store value at start for warm-starting
    num_layers: int = None  # Store which num_layers found the solution
    num_dreams: int = None  # Store which num_dreams found the solution
    iterations: int = None  # Store how many iterations it took
    timing: Dict[str, float] = struct.field(default_factory=dict, pytree_node=False)  # Timing information
    # Store cost matrices for caching (costs with distances, inf for collisions)
    Cs: jax.Array = None
    Ch: jax.Array = None
    Cl: jax.Array = None
    Cg: jax.Array = None


def _snapshot_dream_points(dream_points: jax.Array, max_points: int = 400) -> np.ndarray:
    """Flatten a batched (batch, layers, dreams, dim) dream-point tensor into an
    (N, dim) array of graph nodes, deterministically subsampled to at most
    `max_points` for lightweight per-iteration visualization snapshots."""
    dp = np.asarray(jax.device_get(dream_points))
    dp_flat = dp.reshape(-1, dp.shape[-1])
    if dp_flat.shape[0] > max_points:
        sel = np.linspace(0, dp_flat.shape[0] - 1, max_points).astype(int)
        dp_flat = dp_flat[sel]
    return dp_flat


def expand_dream_points_and_values(
    key: jax.Array,
    bounds: jax.Array,
    prev_dreams: jax.Array,
    prev_Vs: jax.Array,
    prev_Vh: jax.Array,
    prev_m: int,
    prev_n: int,
    new_m: int,
    new_n: int,
    batch_size: int,
    dim: int,
    dtype: jnp.dtype
) -> Tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    """
    Expand dream points and value functions hierarchically.
    
    Args:
        key: JAX random key
        bounds: Configuration space bounds
        prev_dreams: Previous dream points (batch_size, prev_m, prev_n, dim)
        prev_Vs: Previous start values (batch_size,) or scalar
        prev_Vh: Previous dream values (batch_size, prev_m, prev_n)
        prev_m: Previous number of layers
        prev_n: Previous number of dreams
        new_m: New number of layers
        new_n: New number of dreams
        batch_size: Batch size
        dim: Configuration space dimension
        dtype: Data type
    
    Returns:
        Tuple of (new_dreams, Vs_init, Vh_init)
    """
    # Handle layer expansion
    if new_m > prev_m:
        # Consolidate key splits for efficiency
        key, subkey = random.split(key)
        # Efficient key splitting for new layers
        total_new = batch_size * (new_m - prev_m)
        layer_keys = random.split(subkey, total_new)
        layer_keys = layer_keys.reshape(batch_size, new_m - prev_m, 2)
        # vmap over batch and layers
        new_layer_dreams = vmap(lambda ks: vmap(lambda k: sample_dream_points(k, bounds, (1, new_n), dtype=dtype))(ks))(layer_keys)
        new_layer_dreams = new_layer_dreams.squeeze(axis=2)  # Remove singleton dim if needed
        
        # If also expanding dreams in existing layers
        if new_n > prev_n:
            # Expand existing layers with new dreams (reuse same key split)
            additional_dreams = vmap(lambda k: sample_dream_points(
                k, bounds, (prev_m, new_n - prev_n), dtype=dtype
            ))(keys)  # Reuse keys from above
            # Concatenate old and new dreams for existing layers
            existing_layers_expanded = jnp.concatenate([prev_dreams, additional_dreams], axis=2)
            # Combine with new layers
            new_dreams = jnp.concatenate([existing_layers_expanded, new_layer_dreams], axis=1)
        else:
            # Just append new layers, pad existing layers if needed
            if new_n < prev_n:
                existing_layers = prev_dreams[:, :, :new_n, :]
            else:
                existing_layers = prev_dreams
            new_dreams = jnp.concatenate([existing_layers, new_layer_dreams], axis=1)
        
        # Expand Vh using jnp.pad (more efficient than at[].set() which creates copies)
        if prev_Vh is not None:
            min_n = min(prev_n, new_n)
            # Slice to min_n first if needed, then pad
            vh_to_pad = prev_Vh[:, :, :min_n] if prev_n > new_n else prev_Vh
            # Pad dimensions: (batch_size, prev_m, min_n) -> (batch_size, new_m, new_n)
            pad_layers = new_m - prev_m
            pad_dreams = new_n - min_n
            Vh_init = jnp.pad(vh_to_pad, ((0, 0), (0, pad_layers), (0, pad_dreams)), mode='constant', constant_values=0)
        else:
            Vh_init = jnp.zeros((batch_size, new_m, new_n), dtype=dtype)
        Vs_init = prev_Vs
        
    elif new_n > prev_n:
        # Only expanding dreams, not layers
        keys = random.split(key, batch_size)
        additional_dreams = vmap(lambda k: sample_dream_points(
            k, bounds, (new_m, new_n - prev_n), dtype=dtype
        ))(keys)
        # Concatenate old and new dreams
        new_dreams = jnp.concatenate([prev_dreams, additional_dreams], axis=2)
        
        # Expand Vh using jnp.pad (more efficient than at[].set())
        if prev_Vh is not None:
            pad_dreams = new_n - prev_n
            Vh_init = jnp.pad(prev_Vh, ((0, 0), (0, 0), (0, pad_dreams)), mode='constant', constant_values=0)
        else:
            Vh_init = jnp.zeros((batch_size, new_m, new_n), dtype=dtype)
        Vs_init = prev_Vs
        
    else:
        # Same size or smaller (shouldn't happen in typical anytime progression)
        new_dreams = prev_dreams[:, :new_m, :new_n, :]
        Vh_init = prev_Vh[:, :new_m, :new_n] if prev_Vh is not None else None
        Vs_init = prev_Vs
    
    return new_dreams, Vs_init, Vh_init


@jit
def gtmp_plan(key: jax.Array, state: TensorState, 
              dream_points: Optional[jax.Array] = None, Vs_init: Optional[jax.Array] = None, 
              Vh_init: Optional[jax.Array] = None) ->  TensorOutput:
        q = state.q
        batch_size = state.batch_size
        num_layers = state.num_layers
        num_dreams = state.num_dreams
        dim = q.shape[-1]  # Get dim from q shape instead of state.dim
        
        # Sample dream points for all batches (always batched) or use provided ones
        if dream_points is None:
            keys = random.split(key, batch_size)
            dream_points = vmap(lambda k: sample_dream_points(k, state.bounds, (num_layers, num_dreams), dtype=state.dtype))(keys)

        # Assert that q and dream_points have matching last dimension (DoF)
        assert q.shape[-1] == dream_points.shape[-1], (
            f"Shape mismatch: q.shape[-1] = {q.shape[-1]} != dream_points.shape[-1] = {dream_points.shape[-1]}\n"
            f"q.shape={q.shape}, dream_points.shape={dream_points.shape}"
        )
        Cs, Ch, Cl, Cg = state.cost_fn(q, dream_points, state.goals)
        Cs = jnp.asarray(Cs, dtype=state.dtype)
        Ch = jnp.asarray(Ch, dtype=state.dtype)
        Cl = jnp.asarray(Cl, dtype=state.dtype)
        Cg = jnp.asarray(Cg, dtype=state.dtype)

        gamma = 1.0 if state.vi_finite else state.gamma
        gamma = jnp.asarray(gamma, dtype=state.dtype)
        if Vs_init is not None:
            Vs_init = jnp.asarray(Vs_init, dtype=state.dtype)
        if Vh_init is not None:
            Vh_init = jnp.asarray(Vh_init, dtype=state.dtype)
        
        # Value iteration (always batched)
        if num_layers > 1:
            value_iteration_fn = value_iteration_finite if state.vi_finite else value_iteration
            vi_kwargs = {'dtype': state.dtype, 'Vs_init': Vs_init, 'Vh_init': Vh_init}
            if not state.vi_finite:
                vi_kwargs['gamma'] = gamma
            Vs, Vh = value_iteration_fn(Cs, Ch, Cl, Cg, **vi_kwargs)
        else:
            # 1-layer case
            term2 = jnp.min(Cl + gamma * Cg, axis=-1)
            Vs = jnp.min(Cs + gamma * term2, axis=-1)
            Vh = term2[:, None, :]

        collision = jnp.isinf(Vs)
        path_vel = None

        # Path extraction (always batched) - reshape from dream grid
        def get_path_batched(_):
            if num_layers > 1:
                mid_idx, goal_idx = get_optimal_path(Cs, Ch, Cl, Cg, Vh, gamma)
                # Reshape dream_points: (batch_size, num_layers, num_dreams, dim) -> (batch_size * num_layers, num_dreams, dim)
                dream_points_reshaped = dream_points.reshape(batch_size * num_layers, num_dreams, dim)
                # Extract mid_path using flattened indices
                batch_idx = jnp.arange(batch_size)[:, None]  # Shape: (batch_size, 1)
                layer_idx = jnp.arange(num_layers)[None, :]  # Shape: (1, num_layers)
                mid_path = dream_points[batch_idx, layer_idx, mid_idx, :]  # Shape: (batch_size, num_layers, dim)

                goal = state.goals[goal_idx, :, None]  # Shape: (batch_size, dim, 1)
                goal = jnp.moveaxis(goal, -1, 1)  # Shape: (batch_size, 1, dim)

                start = jnp.broadcast_to(q[None, None, :], (batch_size, 1, dim))

                # Concatenate
                path = jnp.concatenate((start, mid_path, goal), axis=1)
            else:
                mid_idx = jnp.argmin(Cs + gamma * Vh[:, 0], axis=-1)
                goal_idx = jnp.argmin(Cl[jnp.arange(batch_size), mid_idx] + gamma * Cg, axis=-1)
                mid_point = dream_points[jnp.arange(batch_size), 0, mid_idx, :]
                goal = state.goals[goal_idx, None, :]
                start = jnp.broadcast_to(q[None, None, :], (batch_size, 1, dim))
                path = jnp.concatenate((start, mid_point[:, None, :], goal), axis=1)
            return path.astype(state.dtype), goal_idx
        
        # Handle collision case
        path, goal_idx = lax.cond(
            jnp.all(collision),
            lambda _: (jnp.zeros((batch_size, num_layers + 2, dim), state.dtype), jnp.zeros(batch_size, dtype=jnp.result_type(int))),
            get_path_batched, None)

        output = TensorOutput(
            path=path, goal_idx=goal_idx, collision=collision, path_vel=path_vel, 
            splines=None, Vs=Vs, dream_points=dream_points, V=Vh,
            Cs=Cs, Ch=Ch, Cl=Cl, Cg=Cg  # Store cost matrices (distances + inf)
        )
        return output


@jit
def gtmp_plan_externally_computed_costs(
    state: TensorState,
    dream_points: jax.Array,
    Cs: jax.Array,
    Ch: jax.Array,
    Cl: jax.Array,
    Cg: jax.Array,
    Vs_init: Optional[jax.Array] = None,
    Vh_init: Optional[jax.Array] = None,
) -> TensorOutput:
    """Run GTMP value iteration + path extraction with externally-supplied
    cost matrices, skipping the call into ``state.cost_fn``.

    This is the path to take when the per-edge costs come from an *outside*
    process (e.g. a sweep of AORRTC edge replanning at varying iteration
    budgets) rather than from the planner's own validator. ``gtmp_plan`` is
    left untouched.

    Expected shapes (match what CostVAMP produces):
        Cs (batch_size, num_dreams)
        Ch (batch_size, num_layers - 1, num_dreams, num_dreams)
        Cl (batch_size, num_dreams, num_goals)
        Cg (num_goals,)
    Infeasible edges should be encoded as ``inf``.
    """
    q = state.q
    batch_size = state.batch_size
    num_layers = state.num_layers
    num_dreams = state.num_dreams
    dim = q.shape[-1]

    Cs = jnp.asarray(Cs, dtype=state.dtype)
    Ch = jnp.asarray(Ch, dtype=state.dtype)
    Cl = jnp.asarray(Cl, dtype=state.dtype)
    Cg = jnp.asarray(Cg, dtype=state.dtype)

    gamma = 1.0 if state.vi_finite else state.gamma
    gamma = jnp.asarray(gamma, dtype=state.dtype)
    if Vs_init is not None:
        Vs_init = jnp.asarray(Vs_init, dtype=state.dtype)
    if Vh_init is not None:
        Vh_init = jnp.asarray(Vh_init, dtype=state.dtype)

    if num_layers > 1:
        value_iteration_fn = value_iteration_finite if state.vi_finite else value_iteration
        vi_kwargs = {'dtype': state.dtype, 'Vs_init': Vs_init, 'Vh_init': Vh_init}
        if not state.vi_finite:
            vi_kwargs['gamma'] = gamma
        Vs, Vh = value_iteration_fn(Cs, Ch, Cl, Cg, **vi_kwargs)
    else:
        term2 = jnp.min(Cl + gamma * Cg, axis=-1)
        Vs = jnp.min(Cs + gamma * term2, axis=-1)
        Vh = term2[:, None, :]

    collision = jnp.isinf(Vs)

    def get_path_batched(_):
        if num_layers > 1:
            mid_idx, goal_idx = get_optimal_path(Cs, Ch, Cl, Cg, Vh, gamma)
            batch_idx = jnp.arange(batch_size)[:, None]
            layer_idx = jnp.arange(num_layers)[None, :]
            mid_path = dream_points[batch_idx, layer_idx, mid_idx, :]
            goal = state.goals[goal_idx, :, None]
            goal = jnp.moveaxis(goal, -1, 1)
            start = jnp.broadcast_to(q[None, None, :], (batch_size, 1, dim))
            path = jnp.concatenate((start, mid_path, goal), axis=1)
        else:
            mid_idx = jnp.argmin(Cs + gamma * Vh[:, 0], axis=-1)
            goal_idx = jnp.argmin(
                Cl[jnp.arange(batch_size), mid_idx] + gamma * Cg, axis=-1,
            )
            mid_point = dream_points[jnp.arange(batch_size), 0, mid_idx, :]
            goal = state.goals[goal_idx, None, :]
            start = jnp.broadcast_to(q[None, None, :], (batch_size, 1, dim))
            path = jnp.concatenate((start, mid_point[:, None, :], goal), axis=1)
        return path.astype(state.dtype), goal_idx

    path, goal_idx = lax.cond(
        jnp.all(collision),
        lambda _: (jnp.zeros((batch_size, num_layers + 2, dim), state.dtype),
                   jnp.zeros(batch_size, dtype=jnp.result_type(int))),
        get_path_batched, None,
    )

    return TensorOutput(
        path=path, goal_idx=goal_idx, collision=collision,
        path_vel=None, splines=None, Vs=Vs, dream_points=dream_points, V=Vh,
        Cs=Cs, Ch=Ch, Cl=Cl, Cg=Cg,
    )


def gtmp_plan_anytime(
    key: jax.Array,
    state: TensorState,
    min_layers: int = 1,
    max_layers: int = 30,
    min_dreams: int = 20,
    max_dreams: int = 200,
    dreams_step: int = 10,
    spline_interpolator: Optional[Any] = None,
    verbose: bool = True,
    time_budget: Optional[float] = None,
    stop_on_first_feasible: bool = False,
    record_samples: bool = False,
) -> TensorOutput:
    """
    Anytime GTMP planner that reruns fresh GTMP solves and keeps the best valid paths.
    Each rerun discards the previous graph and retains up to batch_size
    collision-free solutions ranked by joint-space path length.
    
    Args:
        key: JAX random key
        state: TensorState (num_layers and num_dreams will be overridden)
        min_layers: Number of layers. Held fixed for the run (default: 1)
        max_layers: Unused -- kept for signature compatibility with ao_gtmp_plan
        min_dreams: Number of dream points per layer. Held fixed for the run
            (default: 20)
        max_dreams: Unused -- kept for signature compatibility with ao_gtmp_plan
        dreams_step: Unused -- kept for signature compatibility with ao_gtmp_plan
        spline_interpolator: Optional spline interpolator
        verbose: Print progress (default: True)

    Returns:
        TensorOutput containing up to batch_size accumulated valid solutions
    """
    best_output = None
    iteration = 0
    total_time = 0.0
    start_time = time.perf_counter()
    stats_records = []

    batch_size = state.batch_size
    m = min_layers
    n = min_dreams
    updated_state = state.replace(num_layers=m, num_dreams=n).update_num_layer(m)

    # The graph size is held FIXED at (min_layers, min_dreams) for the whole
    # budget by design: this planner is a random-restart baseline whose whole
    # point is that every restart draws an independent graph at one resolution,
    # which is what gives the retained set its homotopy diversity. Growing the
    # graph mid-run would confound that with an AO-style densification schedule.
    # max_layers/max_dreams/dreams_step are accepted only for signature
    # compatibility with ao_gtmp_plan and are deliberately unused; tune the
    # operating resolution through min_layers/min_dreams instead.

    retained_solutions = []
    # Best collision-free path per iteration, in iteration order.
    # Preserved separately from retained_solutions (which is globally top-k by cost).
    iter_best_paths: list = []

    while True:
        # Check time budget before starting iteration.
        if time_budget is not None and (time.perf_counter() - start_time) >= time_budget:
            if verbose:
                print(f"Time budget of {time_budget}s reached after {iteration} iterations ({total_time:.4f}s total)")
            break

        # Without a time budget, preserve previous behavior and stop once enough
        # solutions are retained.
        if time_budget is None and len(retained_solutions) >= batch_size:
            break

        iteration += 1
        key, iter_seed = random.split(key)
        iter_key = random.fold_in(iter_seed, iteration)

        # Fresh random dreams each iteration invalidate the cost-fn cache,
        # which indexes validation results by (batch, layer, dream_idx).
        if hasattr(state.cost_fn, 'clear_cache'):
            state.cost_fn.clear_cache()

        # Run planning fresh each iteration (no warm-start/cached graph reuse)
        iter_start = time.perf_counter()
        output = gtmp_plan(iter_key, updated_state)
        output.path.block_until_ready()
        iter_time = time.perf_counter() - iter_start
        total_time += iter_time

        # Check if we found any collision-free solutions
        num_collision_free = int(jax.device_get(jnp.sum(~output.collision)))

        collisions = jax.device_get(output.collision)

        # Only retain collision-free candidates; rank by joint-space path length.
        iter_best_cost = np.inf
        iter_best_path = None
        for idx in range(batch_size):
            if bool(collisions[idx]):
                continue
            path_i = jax.device_get(output.path[idx])
            seg_lengths = np.linalg.norm(np.diff(path_i, axis=0), axis=-1)
            path_cost = float(np.sum(seg_lengths))
            if path_cost < iter_best_cost:
                iter_best_cost = path_cost
                iter_best_path = path_i
            retained_solutions.append({
                'collision': False,
                'cost': path_cost,
                'path': path_i,
                'goal_idx': jax.device_get(output.goal_idx[idx]),
                'dream_points': jax.device_get(output.dream_points[idx]),
                'V': jax.device_get(output.V[idx]),
                'Vs': jax.device_get(output.Vs[idx]),
                'Cs': jax.device_get(output.Cs[idx]),
                'Ch': jax.device_get(output.Ch[idx]),
                'Cl': jax.device_get(output.Cl[idx]),
                'Cg': jax.device_get(output.Cg[idx]),
            })

        if iter_best_path is not None:
            entry = {
                'iteration': iteration,
                'cost': iter_best_cost,
                'path': iter_best_path,
            }
            if record_samples:
                entry['dream_points'] = _snapshot_dream_points(output.dream_points)
            iter_best_paths.append(entry)

        if len(retained_solutions) > 0:
            retained_solutions.sort(key=lambda x: x['cost'])
            retained_solutions = retained_solutions[:batch_size]

        # Calculate entropy from collision-free paths
        entropy_val = 0.0
        if num_collision_free > 0:
            # Extract collision-free paths
            computed_paths = jax.device_get(output.path)
            paths = []
            for i in range(batch_size):
                if not collisions[i]:
                    paths.append(computed_paths[i])
            if len(paths) > 0:
                paths_np = np.array(paths)
                entropy_val = entropy_path(paths_np)

        # Collect stats for this iteration
        retained_collision_free = int(sum(not s['collision'] for s in retained_solutions))
        best_cost = np.inf
        for s in retained_solutions:
            if not s['collision']:
                best_cost = s['cost']
                break

        # Record time to first feasible solution
        time_to_first_feasible = total_time if (num_collision_free > 0 and not any(
            r.get('time_to_first_feasible') is not None for r in stats_records
        )) else None

        stats_records.append({
            'iteration': iteration,
            'layers': m,
            'dreams': n,
            'iter_collision_free': num_collision_free,
            'collision_free': num_collision_free,
            'batch_size': state.batch_size,
            'iter_time': iter_time,
            'total_time': total_time,
            'entropy': entropy_val,
            'retained': len(retained_solutions),
            'retained_collision_free': retained_collision_free,
            'iter_best_cost': iter_best_cost,
            'global_best_cost': best_cost,
            'best_cost': best_cost,
            'time_to_first_feasible': time_to_first_feasible,
        })

        timing_info = {
            'iteration_time': iter_time,
            'total_time': total_time,
            'iterations': iteration,
            'collected': len(retained_solutions),
            'target_batch_size': batch_size,
            'iter_best_paths': iter_best_paths,
        }
        best_output = output.replace(
            num_layers=m,
            num_dreams=n,
            iterations=iteration,
            timing=timing_info
        )

        if verbose:
            iter_best_str = f"{iter_best_cost:.6f}" if np.isfinite(iter_best_cost) else "inf (none found)"
            global_best_str = f"{best_cost:.6f}" if np.isfinite(best_cost) else "inf (none found)"
            print(f"Iteration {iteration}: layers={m}, dreams={n}, "
                f"iter_collision_free={num_collision_free}/{state.batch_size}, "
                f"retained_best={len(retained_solutions)}/{batch_size}, "
                f"retained_collision_free={retained_collision_free}/{len(retained_solutions)}, "
                f"iter_best_cost={iter_best_str}, "
                f"global_best_cost={global_best_str}, "
                f"time={iter_time:.4f}s")

        if stop_on_first_feasible and num_collision_free > 0:
            if verbose:
                print(f"First feasible solution found at iteration {iteration}, time={total_time:.4f}s. Stopping.")
            break

    # Build an output from retained solutions (collision-free candidates are
    # prioritized, then shortest path cost).
    if len(retained_solutions) > 0:
        # Callers index the returned batch by range(batch_size), so pad short
        # runs back out to batch_size with collision=True filler drawn from the
        # last iteration's batch. Without this a run that retains 1..batch_size-1
        # solutions returns a short batch and every downstream consumer raises
        # IndexError -- i.e. partially successful runs look like hard failures.
        n_retained = len(retained_solutions)
        n_pad = max(0, batch_size - n_retained)
        if n_pad > 0:
            filler = {
                'path': jax.device_get(output.path[0]),
                'goal_idx': jax.device_get(output.goal_idx[0]),
                'dream_points': jax.device_get(output.dream_points[0]),
                'V': jax.device_get(output.V[0]),
                'Vs': jax.device_get(output.Vs[0]),
                'Cs': jax.device_get(output.Cs[0]),
                'Ch': jax.device_get(output.Ch[0]),
                'Cl': jax.device_get(output.Cl[0]),
                'Cg': jax.device_get(output.Cg[0]),
            }
            retained_solutions = retained_solutions + [
                {'collision': True, 'cost': np.inf, **filler} for _ in range(n_pad)
            ]

        def _stack(field, edge_pad=False):
            """Stack a field across retained solutions, padding ragged shapes.

            With resolution escalation enabled, solutions retained at different
            (layers, dreams) carry graph tensors of different shapes, so a plain
            np.stack raises. Pad each to the elementwise-max shape instead.
            Paths must use ``edge_pad`` -- repeating the final waypoint appends
            zero-length segments, leaving the path's cost and validity untouched,
            whereas zero-padding would splice in a bogus waypoint at the origin.
            """
            arrs = [np.asarray(s[field]) for s in retained_solutions]
            shapes = {a.shape for a in arrs}
            if len(shapes) > 1:
                target = tuple(max(a.shape[d] for a in arrs) for d in range(arrs[0].ndim))
                padded = []
                for a in arrs:
                    if a.shape != target:
                        if edge_pad:
                            pad_width = [(0, t - s) for t, s in zip(target, a.shape)]
                            a = np.pad(a, pad_width, mode='edge')
                        else:
                            buf = np.zeros(target, dtype=a.dtype)
                            buf[tuple(slice(0, s) for s in a.shape)] = a
                            a = buf
                    padded.append(a)
                arrs = padded
            return np.stack(arrs, axis=0)

        path_arr = jnp.asarray(_stack('path', edge_pad=True), dtype=state.dtype)
        goal_idx_arr = jnp.asarray(np.array([s['goal_idx'] for s in retained_solutions]))
        dream_points_arr = jnp.asarray(_stack('dream_points'), dtype=state.dtype)
        V_arr = jnp.asarray(_stack('V'), dtype=state.dtype)
        Vs_arr = jnp.asarray(np.array([s['Vs'] for s in retained_solutions]), dtype=state.dtype)
        Cs_arr = jnp.asarray(_stack('Cs'), dtype=state.dtype)
        Ch_arr = jnp.asarray(_stack('Ch'), dtype=state.dtype)
        Cl_arr = jnp.asarray(_stack('Cl'), dtype=state.dtype)
        Cg_arr = jnp.asarray(_stack('Cg'), dtype=state.dtype)
        collision_arr = jnp.asarray(np.array([s['collision'] for s in retained_solutions]), dtype=bool)

        timing_info = {
            'iteration_time': stats_records[-1]['iter_time'] if len(stats_records) > 0 else 0.0,
            'total_time': total_time,
            'iterations': iteration,
            'collected': n_retained,
            'target_batch_size': batch_size,
            'iter_best_paths': iter_best_paths,
        }
        best_output = best_output.replace(
            path=path_arr,
            goal_idx=goal_idx_arr,
            collision=collision_arr,
            dream_points=dream_points_arr,
            V=V_arr,
            Vs=Vs_arr,
            Cs=Cs_arr,
            Ch=Ch_arr,
            Cl=Cl_arr,
            Cg=Cg_arr,
            num_layers=m,
            num_dreams=n,
            iterations=iteration,
            timing=timing_info
        )

    if verbose:
        n_valid = sum(1 for s in retained_solutions if not s['collision'])
        print(f"Retained best {n_valid}/{batch_size} valid solutions after {iteration} iterations ({total_time:.4f}s total)")

    # Ensure final computation is complete
    if best_output is not None:
        best_output.path.block_until_ready()

    stats_df = pd.DataFrame(stats_records) if verbose else None
    return (best_output, stats_df) if verbose else best_output


def _sample_hyperellipsoid(
    key: jax.Array,
    q: jax.Array,           # (dim,) start config
    goal_cfg: jax.Array,    # (dim,) goal config
    c_best: float,          # cost of best solution found so far
    bounds: jax.Array,      # (dim, 2) configuration space bounds
    num_groups: int,
    num_samples: int,
    batch_size: int,
    dim: int,
    dtype: jnp.dtype,
) -> Tuple[jax.Array, jax.Array]:
    """Sample uniformly from the informed RRT* prolate hyperspheroid.

    The ellipsoid has q and goal_cfg as focal points and transverse diameter
    c_best (the best solution cost found so far).  The eccentricity is
    c_min / c_best, where c_min = ||goal - q|| is the straight-line lower
    bound on path cost, matching the definition from Gammell et al.
    (Informed RRT*, 2014).

    Samples are guaranteed to lie on paths no worse than c_best and are
    clipped to the configuration space bounds.
    """
    c_min = jnp.linalg.norm(goal_cfg - q)
    x_center = (q + goal_cfg) * 0.5

    # Semi-axes of the ellipsoid:
    #   transverse (along goal direction): a = c_best / 2
    #   conjugate  (all orthogonal axes):  b = sqrt(a^2 - (c_min/2)^2)
    a = c_best / 2.0
    b = jnp.sqrt(jnp.maximum(a ** 2 - (c_min * 0.5) ** 2, 1e-8))

    # Rotation matrix C so that C @ e1 = a1 (first axis aligned with goal direction).
    # Uses the SVD trick from Gammell et al. to guarantee det(C) = +1.
    a1 = (goal_cfg - q) / jnp.maximum(c_min, 1e-8)   # unit vector (dim,)
    e1 = jnp.eye(dim)[0]                               # (dim,)
    M = jnp.outer(a1, e1)                              # (dim, dim)
    U, _, Vt = jnp.linalg.svd(M)
    det_sign = jnp.linalg.det(U) * jnp.linalg.det(Vt.T)
    diag_corr = jnp.ones(dim).at[-1].set(det_sign)
    C = U @ jnp.diag(diag_corr) @ Vt                  # proper rotation, C @ e1 = a1

    # Combined linear transform: unit ball -> oriented ellipsoid
    L_diag = jnp.concatenate([jnp.array([a]), jnp.full(dim - 1, b)])
    CL = C @ jnp.diag(L_diag)                         # (dim, dim)

    # Sample uniformly in the unit n-ball via the direction-radius decomposition:
    #   direction ~ uniform on (n-1)-sphere (normalised Gaussian)
    #   radius    ~ uniform^(1/n) (inverse CDF for ball volume)
    key, dir_key, r_key = random.split(key, 3)
    dirs = random.normal(dir_key, (batch_size, num_groups, num_samples, dim))
    dirs = dirs / (jnp.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8)
    r = random.uniform(r_key, (batch_size, num_groups, num_samples, 1)) ** (1.0 / dim)
    x_ball = dirs * r                                  # (B, G, num_samples, dim)

    # Affine transform into the ellipsoid frame and clip to config space bounds
    x_rand = jnp.einsum('ij,...j->...i', CL, x_ball) + x_center
    x_rand = jnp.clip(x_rand, bounds[:, 0], bounds[:, 1])

    return x_rand.astype(dtype), key


def expand_dream_points_analytical(
    key: jax.Array,
    bounds: jax.Array,
    prev_dreams: jax.Array,
    prev_Vs: jax.Array,
    prev_Vh: jax.Array,
    prev_m: int,
    prev_n: int,
    new_m: int,
    new_n: int,
    batch_size: int,
    dim: int,
    dtype: jnp.dtype,
    q: jax.Array,
    goals: jax.Array,
    c_best: float = np.inf,
    explore_frac: float = 0.5,
    prev_Cs: Optional[jax.Array] = None,
    prev_Ch: Optional[jax.Array] = None,
    prev_Cl: Optional[jax.Array] = None,
    prev_Cg: Optional[jax.Array] = None,
) -> Tuple[jax.Array, Optional[jax.Array], Optional[jax.Array], jax.Array]:
    """Expand dream points with a 50/50 mix of new samples and analytical midpoints.

    A fraction ``explore_frac`` (default 0.5) of each expansion is a *new*
    sample; the remainder are analytical midpoints of the shortest source-target
    pair in that layer (triangle-inequality optimal: d(s,x)+d(x,t) = d(s,t) at
    the midpoint).

    New samples come from the informed RRT* prolate hyperspheroid defined by q,
    goal, and c_best (eccentricity = c_min / c_best) once c_best is finite, and
    uniformly from the configuration space before then -- the c_best = inf limit
    of that same ellipsoid.
    """
    goal_cfg = goals[0] if goals.ndim > 1 else goals
    use_ellipsoid = np.isfinite(c_best)

    def build_adjacent_refs(layer_points: jax.Array, num_layers: int, num_points: int) -> Tuple[jax.Array, jax.Array]:
        if num_layers <= 0:
            empty = jnp.zeros((batch_size, 0, 1, dim), dtype=dtype)
            return empty, empty

        ref_count = max(1, num_points)
        start_ref = jnp.broadcast_to(q[None, None, None, :], (batch_size, 1, ref_count, dim))
        goal_ref = jnp.broadcast_to(goal_cfg[None, None, None, :], (batch_size, 1, ref_count, dim))

        if num_layers == 1:
            return start_ref, goal_ref

        left_layers = layer_points[:, :num_layers - 1, :ref_count, :]
        right_layers = layer_points[:, 1:num_layers, :ref_count, :]
        source = jnp.concatenate([start_ref, left_layers], axis=1)
        target = jnp.concatenate([right_layers, goal_ref], axis=1)
        return source, target

    def best_midpoints(
        source_refs: jax.Array,  # (B, num_groups, k_s, dim)
        target_refs: jax.Array,  # (B, num_groups, k_t, dim)
        num_groups: int,
        num_samples: int,
    ) -> jax.Array:
        """Return midpoints of the num_samples shortest source-target pairs per group."""
        if num_groups <= 0:
            return jnp.zeros((batch_size, 0, num_samples, dim), dtype=dtype)

        k_s = source_refs.shape[2]
        k_t = target_refs.shape[2]

        diffs = source_refs[:, :, :, None, :] - target_refs[:, :, None, :, :]
        pairwise_dist = jnp.linalg.norm(diffs, axis=-1)

        num_pairs = k_s * k_t
        dist_flat = pairwise_dist.reshape(batch_size, num_groups, num_pairs)
        num_select = min(num_samples, num_pairs)
        top_idx = jnp.argsort(dist_flat, axis=-1)[:, :, :num_select]

        s_idx = top_idx // k_t
        t_idx = top_idx % k_t

        bidx = jnp.arange(batch_size)[:, None, None]
        gidx = jnp.arange(num_groups)[None, :, None]
        src_pts = source_refs[bidx, gidx, s_idx, :]
        tgt_pts = target_refs[bidx, gidx, t_idx, :]
        midpoints = (src_pts + tgt_pts) * 0.5

        if num_select < num_samples:
            pad = jnp.broadcast_to(
                midpoints[:, :, -1:, :],
                (batch_size, num_groups, num_samples - num_select, dim),
            )
            midpoints = jnp.concatenate([midpoints, pad], axis=2)

        return midpoints.astype(dtype)

    def mixed_samples(
        source_refs: jax.Array,
        target_refs: jax.Array,
        num_groups: int,
        num_samples: int,
    ) -> jax.Array:
        """Split each expansion between analytical midpoints and new samples.

        ``explore_frac`` of the batch is *new* samples and the rest is analytical
        triangle-inequality midpoints (default 50/50).

        The new-sample half is drawn from the informed hyperellipsoid once a
        finite ``c_best`` exists. Before that it is drawn uniformly from the
        configuration space -- which is the correct limit, since the informed
        ellipsoid with c_best = inf *is* the whole space.

        This is the fix for the pre-solution collapse: the split used to be
        gated on ``use_ellipsoid``, so with no incumbent yet the new-sample half
        silently fell back to midpoints and the expansion became 100%
        exploitative. Midpoints only ever land between existing pairs, so at
        that point the graph could no longer discover an unvisited passage --
        exactly when it most needed to.
        """
        nonlocal key
        if num_groups <= 0:
            return best_midpoints(source_refs, target_refs, num_groups, num_samples)

        n_new = int(round(num_samples * explore_frac))
        n_new = min(max(n_new, 0), num_samples)
        n_analytical = num_samples - n_new

        parts = []
        if n_analytical > 0:
            parts.append(best_midpoints(source_refs, target_refs, num_groups, n_analytical))

        if n_new > 0:
            if use_ellipsoid:
                new_samples, key = _sample_hyperellipsoid(
                    key, q, goal_cfg, c_best, bounds,
                    num_groups, n_new, batch_size, dim, dtype,
                )
            else:
                key, uni_key = random.split(key)
                new_samples = random.uniform(
                    uni_key, (batch_size, num_groups, n_new, dim),
                    minval=bounds[:, 0], maxval=bounds[:, 1],
                ).astype(dtype)
            parts.append(new_samples)

        return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=2)

    add_layers = max(0, new_m - prev_m)
    add_dreams = max(0, new_n - prev_n)

    # Place new layer samples: 50% analytical midpoints, 50% hyperellipsoid.
    if add_layers > 0:
        if prev_m > 0 and prev_n > 0:
            source_new = prev_dreams[:, prev_m - 1:prev_m, :prev_n, :]
        else:
            source_new = jnp.broadcast_to(q[None, None, None, :], (batch_size, 1, 1, dim))
        target_new = jnp.broadcast_to(goal_cfg[None, None, None, :], (batch_size, 1, 1, dim))
        source_new = jnp.broadcast_to(source_new, (batch_size, add_layers, source_new.shape[2], dim))
        target_new = jnp.broadcast_to(target_new, (batch_size, add_layers, target_new.shape[2], dim))
        new_layer_dreams = mixed_samples(source_new, target_new, add_layers, new_n)
    else:
        new_layer_dreams = jnp.zeros((batch_size, 0, new_n, dim), dtype=dtype)

    # Expand existing layers: 50% analytical midpoints, 50% hyperellipsoid.
    if add_dreams > 0 and prev_m > 0:
        source_existing, target_existing = build_adjacent_refs(
            prev_dreams[:, :prev_m, :prev_n, :], prev_m, prev_n
        )
        additional_dreams = mixed_samples(source_existing, target_existing, prev_m, add_dreams)
        existing_layers_expanded = jnp.concatenate([prev_dreams, additional_dreams], axis=2)
    else:
        existing_layers_expanded = prev_dreams[:, :, :new_n, :]

    # Assemble final dream tensor
    if add_layers > 0:
        if prev_m > 0:
            new_dreams = jnp.concatenate([existing_layers_expanded, new_layer_dreams], axis=1)
        else:
            new_dreams = new_layer_dreams
    else:
        new_dreams = existing_layers_expanded[:, :new_m, :new_n, :]

    # Warm-start value tensors with adjacent-layer cost-to-come + cost-to-go.
    source_full, target_full = build_adjacent_refs(new_dreams[:, :new_m, :new_n, :], new_m, new_n)
    cost_to_come_full = jnp.min(
        jnp.linalg.norm(new_dreams[:, :new_m, :new_n, None, :] - source_full[:, :, None, :, :], axis=-1),
        axis=-1,
    )
    cost_to_go_full = jnp.min(
        jnp.linalg.norm(new_dreams[:, :new_m, :new_n, None, :] - target_full[:, :, None, :, :], axis=-1),
        axis=-1,
    )
    Vh_heuristic = (cost_to_come_full + cost_to_go_full).astype(dtype)
    if prev_Vh is not None:
        keep_m = min(prev_m, new_m)
        keep_n = min(prev_n, new_n)
        Vh_init = Vh_heuristic.at[:, :keep_m, :keep_n].set(prev_Vh[:, :keep_m, :keep_n])
    else:
        Vh_init = Vh_heuristic

    Vs_init = prev_Vs
    return new_dreams, Vs_init, Vh_init, key



def ao_gtmp_plan(
    key: jax.Array,
    state: TensorState,
    min_layers: int = 1,
    max_layers: int = 30,
    min_dreams: int = 20,
    max_dreams: int = 200,
    dreams_step: int = 10,
    explore_frac: float = 0.5,
    spline_interpolator: Optional[Any] = None,
    verbose: bool = True,
    time_budget: Optional[float] = None,
    stop_on_first_feasible: bool = False,
    record_samples: bool = False,
) -> TensorOutput:
    """
    AOGTMP planner that iteratively increases graph complexity and uses the
    triangle inequality to analytically place new dream points in promising
    regions of the state space.  New samples are placed at the midpoints of
    the source-target pairs with the smallest pairwise distance, guaranteeing
    a heuristic of d(s, t) — the minimum achievable by the triangle inequality.
    Previous dream points are reused and value iteration is warm-started.

    Args:
        key: JAX random key
        state: TensorState (num_layers and num_dreams will be overridden)
        min_layers: Minimum number of layers to try (default: 1)
        max_layers: Maximum number of layers to try (default: 30)
        min_dreams: Minimum number of dream points to try (default: 20)
        max_dreams: Maximum number of dream points to try (default: 200)
        dreams_step: Step size for incrementing dreams (default: 10)
        explore_frac: Fraction of each expansion drawn as new samples (informed
            ellipsoid once an incumbent exists, uniform before then) rather than
            as analytical triangle-inequality midpoints (default: 0.5)
        spline_interpolator: Optional spline interpolator
        verbose: Print progress (default: True)
        stop_on_first_feasible: Return as soon as an iteration produces any
            collision-free path, instead of running the full anytime schedule
            (or the time budget). Useful for time-to-first-solution benchmarks
            where post-solution optimization is not measured. (default: False)

    Returns:
        TensorOutput for the best incumbent found during the anytime schedule,
        or the last attempt if no feasible path is found.
    """
    best_output = None
    best_feasible_cost = np.inf
    best_joint_length = np.inf
    best_m = min_layers
    best_n = min_dreams
    best_iteration = 0
    iteration = 0
    total_time = 0.0
    start_time = time.perf_counter()
    stats_records = []
    # Per-iteration incumbent history for visualization of informed-sampling progress.
    incumbent_path = None
    # Batch output from the iteration that produced the current incumbent.
    incumbent_output = None
    iter_best_paths: list = []

    # Cache for warm-starting
    prev_dream_points = None
    prev_Vs = None
    prev_Vh = None
    prev_Cs = None
    prev_Ch = None
    prev_Cl = None
    prev_Cg = None
    prev_m = 0
    prev_n = 0

    batch_size = state.batch_size
    dim = state.q.shape[-1]

    m = min_layers
    n = min_dreams

    # Optional edge validator for post-hoc path simplification.
    edge_validator = None
    if hasattr(state.cost_fn, '_batch_edge_validation'):
        def _edge_validator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            if len(a) == 0:
                return np.zeros((0,), dtype=bool)
            return np.asarray(state.cost_fn._batch_edge_validation(a, b), dtype=bool)

        edge_validator = _edge_validator
    elif verbose:
        print("Warning: cost_fn has no _batch_edge_validation; skipping path simplification.")

    while m <= max_layers or n <= max_dreams:
        # Two distinct quantities that must not be conflated:
        #
        #  * cost_bound  -- the per-edge pruning threshold handed to the cost
        #    function. Any edge whose connector cost already exceeds the whole
        #    incumbent path cost cannot lie on a better path, so this shrinks
        #    monotonically with the incumbent and is what makes pruning bite.
        #
        #  * ellipsoid_c_best -- the transverse diameter of the informed
        #    hyperellipsoid. This must not under-cover the incumbent, so the
        #    worst single connector seen is folded in here as a safety margin
        #    (RRTC connectors wiggle around obstacles, so the Euclidean sum
        #    underestimates the true path length).
        #
        # Previously both used max(incumbent, running-max connector cost). That
        # running max never decreases, so the pruning threshold could never
        # shrink below the worst connector ever validated and the bound was
        # effectively inert.
        cost_bound = best_feasible_cost
        if np.isfinite(cost_bound):
            cost_bound = float(np.ceil(cost_bound))

        ellipsoid_c_best = best_feasible_cost
        if hasattr(state.cost_fn, 'get_max_connector_cost'):
            connector_cost = state.cost_fn.get_max_connector_cost()
            if np.isfinite(connector_cost):
                if not np.isfinite(ellipsoid_c_best) or connector_cost > ellipsoid_c_best:
                    ellipsoid_c_best = connector_cost
        if np.isfinite(ellipsoid_c_best):
            ellipsoid_c_best = float(np.ceil(ellipsoid_c_best))

        if hasattr(state.cost_fn, 'set_cost_bound'):
            state.cost_fn.set_cost_bound(cost_bound)

        # Check time budget before starting iteration
        if time_budget is not None and (time.perf_counter() - start_time) >= time_budget:
            if verbose:
                print(f"Time budget of {time_budget}s reached after {iteration} iterations ({total_time:.4f}s total)")
            if best_output is not None:
                best_output.path.block_until_ready()
            stats_df = pd.DataFrame(stats_records) if verbose else None
            return (best_output, stats_df) if verbose else best_output

        iteration += 1
        key, iter_seed = random.split(key)
        iter_key = random.fold_in(iter_seed, iteration)
        sample_key, plan_key = random.split(iter_key)

        iter_start = time.perf_counter()
        sample_start = iter_start

        # Expand dream points hierarchically
        if prev_dream_points is None:
            # First iteration: sample all dream points
            keys = random.split(sample_key, batch_size)
            dream_points = vmap(lambda k: sample_dream_points(k, state.bounds, (m, n), dtype=state.dtype))(keys)
            Vs_init = None
            Vh_init = None
        else:
            # Expand from previous dream points: 50% analytical midpoints + 50% hyperellipsoid
            dream_points, Vs_init, Vh_init, _ = expand_dream_points_analytical(
                sample_key, state.bounds, prev_dream_points, prev_Vs, prev_Vh,
                prev_m, prev_n, m, n, batch_size, dim, state.dtype,
                state.q, state.goals,
                c_best=ellipsoid_c_best,
                explore_frac=explore_frac,
                prev_Cs=prev_Cs,
                prev_Ch=prev_Ch,
                prev_Cl=prev_Cl,
                prev_Cg=prev_Cg,
            )

        # Materialize sampling before timing/accounting to include it explicitly.
        dream_points.block_until_ready()
        sample_time = time.perf_counter() - sample_start
        projected_total = total_time + sample_time

        # Budget accounting includes sampling. Stop before planning if exhausted.
        if time_budget is not None and projected_total >= time_budget:
            total_time = projected_total
            if verbose:
                print(
                    f"Time budget of {time_budget}s reached after sampling on iteration {iteration} "
                    f"({total_time:.4f}s total). Skipping planning step."
                )
            break

        # Update state with new parameters and cached cost validations
        updated_state = state.replace(num_layers=m, num_dreams=n).update_num_layer(m)

        # Do not update cost function cache from TensorState; cache is managed internally by cost function

        # Run planning with timing and warm-start
        plan_start = time.perf_counter()
        output = gtmp_plan(plan_key, updated_state, 
                        dream_points=dream_points, Vs_init=Vs_init, Vh_init=Vh_init)

        # Force device execution so timing reflects true wall-clock cost.
        output.path.block_until_ready()
        plan_time = time.perf_counter() - plan_start
        iter_time = sample_time + plan_time
        total_time += iter_time

        # Cache for next iteration (including cost validations)
        prev_dream_points = output.dream_points
        prev_Vs = output.Vs
        prev_Vh = output.V
        prev_Cs = output.Cs
        prev_Ch = output.Ch
        prev_Cl = output.Cl
        prev_Cg = output.Cg
        prev_m = m
        prev_n = n

        # Check if we found any collision-free solutions
        num_collision_free = jnp.sum(~output.collision)
        iter_best_joint_length = np.inf

        # Simplify all currently optimal batched paths before reporting length.
        collisions_np = np.asarray(jax.device_get(output.collision))
        if edge_validator is not None and np.any(~collisions_np):
            paths_np = np.asarray(jax.device_get(output.path))
            simplify_mask = ~collisions_np
            simplified_paths_np, _ = simplify_paths_batched(
                paths_np,
                simplify_mask,
                edge_validator=edge_validator,
                num_shortcut_rounds=64,
                seed=iteration,
            )
            output = output.replace(path=jnp.asarray(simplified_paths_np, dtype=state.dtype))

        # Calculate entropy from collision-free paths
        entropy_val = 0.0
        if num_collision_free > 0:
            # Extract collision-free paths
            collisions = collisions_np
            computed_paths = jax.device_get(output.path)
            paths = []
            for i in range(batch_size):
                if not collisions[i]:
                    paths.append(computed_paths[i])
            if len(paths) > 0:
                paths_np = np.array(paths)
                entropy_val = entropy_path(paths_np)

        # Record time to first feasible solution (first iteration with a
        # collision-free path), mirroring gtmp_plan_anytime's stats schema.
        time_to_first_feasible = total_time if (int(jax.device_get(num_collision_free)) > 0 and not any(
            r.get('time_to_first_feasible') is not None for r in stats_records
        )) else None

        # Collect stats for this iteration
        stats_records.append({
            'iteration': iteration,
            'layers': m,
            'dreams': n,
            'collision_free': num_collision_free,
            'batch_size': state.batch_size,
            'sample_time': sample_time,
            'plan_time': plan_time,
            'iter_time': iter_time,
            'total_time': total_time,
            'entropy': entropy_val,
            'time_to_first_feasible': time_to_first_feasible,
        })

        # Store the output with metadata and timing
        timing_info = {
            'iteration_time': iter_time,
            'total_time': total_time,
            'iterations': iteration,
            'iter_best_paths': iter_best_paths,
        }
        output = output.replace(
            num_layers=m, 
            num_dreams=n, 
            iterations=iteration,
            timing=timing_info
        )
        # Keep the latest grown-graph batch output for downstream reporting. If
        # an earlier iteration produced a better incumbent, the snapshot taken
        # below wins -- see the `best_output = incumbent_output` restore at the
        # end of the loop body.
        best_output = output

        # Update incumbent with the best feasible candidate from this iteration.
        iter_best_path_np = None
        if int(jax.device_get(num_collision_free)) > 0:
            collisions_np = np.asarray(jax.device_get(output.collision))
            paths_np = np.asarray(jax.device_get(output.path))
            feasible_idx = np.where(~collisions_np)[0]
            feasible_paths = paths_np[feasible_idx]
            seg_lengths = np.linalg.norm(np.diff(feasible_paths, axis=1), axis=-1)
            feasible_costs = np.sum(seg_lengths, axis=1)
            local_best_pos = int(np.argmin(feasible_costs))
            iter_best_path_np = np.asarray(feasible_paths[local_best_pos])

            # The Euclidean sum is a lower bound — actual RRTC connectors wiggle
            # around obstacles, so the true path cost can be higher. Use it for
            # the cost bound so the hyperellipsoid actually contains the path.
            if hasattr(state.cost_fn, 'compute_path_cost'):
                iter_best_cost = float(state.cost_fn.compute_path_cost(iter_best_path_np))
                if not np.isfinite(iter_best_cost):
                    iter_best_cost = float(feasible_costs[local_best_pos])
            else:
                iter_best_cost = float(feasible_costs[local_best_pos])
            iter_best_joint_length = iter_best_cost
            if iter_best_cost < best_feasible_cost:
                best_feasible_cost = iter_best_cost
                best_joint_length = iter_best_cost
                best_m = m
                best_n = n
                best_iteration = iteration
                incumbent_path = iter_best_path_np
                incumbent_output = output
                if verbose:
                    print(
                        f"New incumbent collision-free joint length={best_joint_length:.6f} "
                        f"at iter={iteration}, layers={m}, dreams={n}"
                    )

        # Report the incumbent, not whatever the newest graph happened to
        # produce. Growing the graph (and tightening the cost bound) can leave a
        # later iteration with no collision-free path in its batch; returning
        # that batch would report a solved run as a failure and discard the
        # incumbent entirely.
        if incumbent_output is not None:
            best_output = incumbent_output

        iter_c_best = best_feasible_cost
        if hasattr(state.cost_fn, 'get_max_connector_cost'):
            connector_cost = state.cost_fn.get_max_connector_cost()
            if np.isfinite(connector_cost):
                if not np.isfinite(iter_c_best) or connector_cost > iter_c_best:
                    iter_c_best = connector_cost

        if np.isfinite(iter_c_best):
            iter_c_best = float(np.ceil(iter_c_best))

        iter_entry = {
            'iteration': iteration,
            'c_best': iter_c_best,
            'incumbent_path': None if incumbent_path is None else np.asarray(incumbent_path),
            'iter_best_cost': iter_best_joint_length,
            'iter_best_path': iter_best_path_np,
            'layers': m,
            'dreams': n,
        }
        if record_samples:
            iter_entry['dream_points'] = _snapshot_dream_points(output.dream_points)
        iter_best_paths.append(iter_entry)

        if verbose:
            iter_best_str = f"{iter_best_joint_length:.6f}" if np.isfinite(iter_best_joint_length) else "inf (none found)"
            global_best_str = f"{best_feasible_cost:.6f}" if np.isfinite(best_feasible_cost) else "inf (none found)"
            print(
                f"Iteration {iteration}: layers={m}, dreams={n}, "
                f"collision_free={num_collision_free}/{state.batch_size}, "
                f"iter_best_cost={iter_best_str}, global_best_cost={global_best_str}, "
                f"sample={sample_time:.4f}s, plan={plan_time:.4f}s, total={iter_time:.4f}s"
            )

        # Escape as soon as a feasible solution exists when requested, so
        # time-to-first-solution benchmarks don't pay for further optimization.
        if stop_on_first_feasible and int(jax.device_get(num_collision_free)) > 0:
            if verbose:
                print(f"First feasible solution found at iteration {iteration}, time={total_time:.4f}s. Stopping.")
            break

        # Decide whether to increase layers or dreams to keep it square-ish
        # Calculate how far each dimension has progressed (0 to 1)
        layers_progress = (m - min_layers) / max(1, (max_layers - min_layers))
        dreams_progress = (n - min_dreams) / max(1, (max_dreams - min_dreams))
        
        # Increase the dimension that's lagging behind
        # If both are maxed out, break
        if m >= max_layers and n >= max_dreams:
            break
        elif m >= max_layers:
            # Can only increase dreams
            n += dreams_step
        elif n >= max_dreams:
            # Can only increase layers
            m += 1
        elif layers_progress < dreams_progress:
            # Layers are lagging, increase them
            m += 1
        else:
            # Dreams are lagging (or equal), increase them
            n += dreams_step
        



    if verbose:
        if np.isfinite(best_feasible_cost):
            print(
                f"Finished AO schedule after {iteration} iterations ({total_time:.4f}s total), "
                f"best_feasible_cost={best_feasible_cost:.6f}"
            )
            print(
                f"Best collision-free joint length={best_joint_length:.6f} "
                f"(iter={best_iteration}, layers={best_m}, dreams={best_n})"
            )
        else:
            print(f"No solution found after {iteration} iterations ({total_time:.4f}s total)")

    # Ensure final computation is complete
    if best_output is not None:
        best_output.path.block_until_ready()

    stats_df = pd.DataFrame(stats_records) if verbose else None
    return (best_output, stats_df) if verbose else best_output

