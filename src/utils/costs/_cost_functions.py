

from abc import ABC, abstractmethod
from typing import Tuple, Union, Any, Dict, Callable, Sequence, Optional
import time
from flax import struct
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
from jax import pure_callback
from jax.typing import ArrayLike
import numpy as np
import numpy as onp
import pyroffi as pk
from pyroffi import Robot  
from pyroffi.collision import RobotCollision, CollGeom
from pyroffi.collision._robot_collision import RobotCollisionSpherized  
# from pyroffi.examples.pyroffi_snippets import _solve_ik, _trajopt, _online_planning
try:
    from joblib import Parallel, delayed, parallel_config
except ImportError:
    Parallel = delayed = parallel_config = None
import vamp
import importlib
from loguru import logger
logger.remove()  

@struct.dataclass
class CostVAMP:
    """VAMP-based cost function with mutable cache stored separately."""
    env: Any = struct.field(default=None, pytree_node=False)
    robot_name: str = struct.field(default='panda', pytree_node=False)
    robot_validate_fn: Callable = struct.field(default=None, pytree_node=False)
    n_jobs: int = struct.field(default=1, pytree_node=False)
    validation_method: str = struct.field(default='validate_motion', pytree_node=False)
    validation_settings: Any = struct.field(default=None, pytree_node=False)
    robot_module: Any = struct.field(default=None, pytree_node=False)
    
    # Store cache in a mutable container (dict wrapped in a class)
    _cache_container: Any = struct.field(default=None, pytree_node=False)
    _cache_enabled: bool = struct.field(default=False, pytree_node=False)
    
    @classmethod
    def create(cls, env: Any, robot_name: str = 'panda', n_jobs: int = 1, 
               cache_enabled: bool = True, validation_method: str = 'validate_motion', 
               validation_max_iterations: int = 1000, aorrtc_optimize: bool = True) -> "CostVAMP":
        """Creates a VAMP-based cost function."""
        
        
        valid_methods = ['validate_motion', 'rrtc', 'aorrtc']
        if validation_method not in valid_methods:
            raise ValueError(f"validation_method must be one of {valid_methods}, got '{validation_method}'")
        
        robot_module = getattr(vamp, robot_name)
        robot_validate_fn = robot_module.validate_motion
        print("Validation max iterations:", validation_max_iterations)
        validation_settings = None
        if validation_method == 'rrtc':
            validation_settings = vamp.AORRTCSettings()
            validation_settings.max_iterations = validation_max_iterations
            validation_settings.max_internal_iterations = validation_max_iterations
            validation_settings.max_samples = validation_max_iterations
            validation_settings.optimize = True
            validation_settings.cost_bound_resample = True
            validation_settings.simplify_intermediate = False
            validation_settings.use_phs = True
            validation_settings.anytime = False
            validation_settings.rrtc.max_iterations = validation_max_iterations
            validation_settings.rrtc.max_samples = validation_max_iterations
        elif validation_method == 'aorrtc':
            validation_settings = vamp.AORRTCSettings()
            validation_settings.max_iterations = validation_max_iterations
            validation_settings.optimize = aorrtc_optimize
        
        # Create mutable cache container
        cache_container = {
            'free_s_1': None,
            'free_layers': None,
            'free_final_g': None,
            'prev_dream_points': None,
            'prev_q': None,
            'prev_num_layers': 0,
            'prev_num_dreams': 0,
            'cache_id': 0,
            'cost_bound': np.inf,
            'max_connector_cost': np.nan,
        } if cache_enabled else None
        
        return cls(
            env=env,
            robot_name=robot_name,
            robot_validate_fn=robot_validate_fn,
            n_jobs=n_jobs,
            _cache_enabled=cache_enabled,
            _cache_container=cache_container,
            validation_method=validation_method,
            validation_settings=validation_settings,
            robot_module=robot_module
        )
    
    def clear_cache(self) -> "CostVAMP":
        """Clear all cached validation results."""
        if self._cache_enabled and self._cache_container is not None:
            self._cache_container.update({
                'free_s_1': None,
                'free_layers': None,
                'free_final_g': None,
                'prev_dream_points': None,
                'prev_q': None,
                'prev_num_layers': 0,
                'prev_num_dreams': 0,
                'cache_id': 0,
                'cost_bound': np.inf,
                'max_connector_cost': np.nan,
            })
        return self

    def set_cost_bound(self, cost_bound: Optional[float]) -> None:
        """Update the shared cost bound for validation backends."""
        bound = np.inf if cost_bound is None or not np.isfinite(cost_bound) else float(cost_bound)
        if self._cache_container is not None:
            self._cache_container['cost_bound'] = bound

        if self.validation_settings is None:
            return

        if hasattr(self.validation_settings, 'max_cost'):
            self.validation_settings.max_cost = bound

        if hasattr(self.validation_settings, 'rrtc') and hasattr(self.validation_settings.rrtc, 'max_cost'):
            self.validation_settings.rrtc.max_cost = bound

    def _get_cost_bound(self) -> float:
        if self._cache_container is None:
            return np.inf
        return float(self._cache_container.get('cost_bound', np.inf))

    def get_max_connector_cost(self) -> float:
        if self._cache_container is None:
            return np.nan
        return float(self._cache_container.get('max_connector_cost', np.nan))

    def _update_max_connector_cost(self, batch_results) -> None:
        if self._cache_container is None:
            return

        current = self._cache_container.get('max_connector_cost', np.nan)
        max_cost = current if np.isfinite(current) else -np.inf
        for result in batch_results:
            if not result.solved:
                continue
            cost = result.path.cost()
            if cost > max_cost:
                max_cost = cost

        if max_cost > -np.inf:
            self._cache_container['max_connector_cost'] = max_cost

    def compute_path_cost(self, path: np.ndarray) -> float:
        """Sum of actual connector costs along consecutive waypoints.

        For RRTC/AORRTC, returns the sum of `r.path.cost()` after replanning
        each edge — this captures the true physical path length including any
        wiggling around obstacles, which the Euclidean sum underestimates.
        For straight-line validation, returns the Euclidean sum.
        Returns np.inf if any edge fails to solve.
        """
        path = np.asarray(path, dtype=np.float32)
        if path.shape[0] < 2:
            return 0.0

        pairs_a = np.ascontiguousarray(path[:-1])
        pairs_b = np.ascontiguousarray(path[1:])

        if self.validation_method in ('rrtc', 'aorrtc') and hasattr(self.robot_module, 'aorrtc_batch'):
            batch_results = self.robot_module.aorrtc_batch(
                pairs_a, pairs_b, self.env, self.validation_settings
            )
            total = 0.0
            for r in batch_results:
                if not r.solved:
                    return float('inf')
                total += float(r.path.cost())
            return float(total)

        return float(np.sum(np.linalg.norm(pairs_b - pairs_a, axis=-1)))


    def _vamp_validate_callback(
        self,
        q: np.ndarray,
        dream_points: np.ndarray,
        goals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Callback to run VAMP validation in Python (NumPy)."""
        batch_size = dream_points.shape[0]
        num_layers = dream_points.shape[1]
        num_dreams = dream_points.shape[2]
        num_goals = goals.shape[0]
        
        cache = self._cache_container if self._cache_enabled else None

        # Check if we can use cached results
        can_use_cache = (
            cache is not None and
            cache['cache_id'] > 0 and
            cache['free_s_1'] is not None and
            (num_layers >= cache['prev_num_layers'] or
            num_dreams >= cache['prev_num_dreams'])
        )
        
        # print("DEBUG:", can_use_cache)
        if can_use_cache:
            free_s_1, free_layers, free_final_g = self._incremental_validate(
                q, dream_points, goals,
                batch_size, num_layers, num_dreams, num_goals
            )
        else:
            free_s_1, free_layers, free_final_g = self._full_validate(
                q, dream_points, goals,
                batch_size, num_layers, num_dreams, num_goals
            )
        
        # NOTE: Don't update cache here - it's already updated in _incremental_validate
        # Just update the metadata
        if cache is not None:
            cache['prev_dream_points'] = dream_points.copy()
            cache['prev_q'] = q.copy()
            cache['prev_num_layers'] = num_layers
            cache['prev_num_dreams'] = num_dreams
            cache['cache_id'] += 1
        
        return free_s_1, free_layers, free_final_g
    def _full_validate(
        self,
        q: np.ndarray,
        dream_points: np.ndarray,
        goals: np.ndarray,
        batch_size: int,
        num_layers: int,
        num_dreams: int,
        num_goals: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full validation of all edges (no caching reuse, but stores results in cache).
        Vectorized using NumPy operations.
        """
        cache = self._cache_container
        
        pairs_a = []
        pairs_b = []
        # pair_indices is still needed to reconstruct the result arrays
    
        # pairs_a: q tiled (B * N_d) times
        q_tiled = np.tile(q, (batch_size * num_dreams, 1))
        pairs_a.append(q_tiled)
        
        # pairs_b: dream_points for Layer 0, flattened across B and N_d
        l0_dreams_flat = dream_points[:, 0, :].reshape(-1, dream_points.shape[-1])
        pairs_b.append(l0_dreams_flat)
        
        if num_layers > 1:
            # dream_points: (B, N_l, N_d, dim)
            # We want all N_d x N_d pairs for each (B, L)
            
            # dreams_l: (B, N_l-1, N_d, dim) - all layers except the last one
            dreams_l = dream_points[:, :-1]
            # dreams_l_plus_1: (B, N_l-1, N_d, dim) - all layers except the first one
            dreams_l_plus_1 = dream_points[:, 1:]
            
            # Reshape for broadcasting. L is the layer index (0 to N_l-2)
            # dreams_l_A: (B, N_l-1, N_d, 1, dim). Repeat N_d times on axis 3
            dreams_l_A = np.repeat(dreams_l[:, :, :, np.newaxis, :], num_dreams, axis=3)
            # dreams_l_B: (B, N_l-1, 1, N_d, dim). Repeat N_d times on axis 2
            dreams_l_B = np.tile(dreams_l_plus_1[:, :, np.newaxis, :, :], (1, 1, num_dreams, 1, 1))
            
            # Flatten to (B * (N_l-1) * N_d^2, dim)
            pairs_a.append(dreams_l_A.reshape(-1, dream_points.shape[-1]))
            pairs_b.append(dreams_l_B.reshape(-1, dream_points.shape[-1]))

        # --- 3. Last Layer -> Goals (across all batches) ---
        # last_layer_dreams: (B, N_d, dim). Tile across N_g
        last_layer_dreams = dream_points[:, -1, :]
        dreams_G_A = np.repeat(last_layer_dreams[:, np.newaxis, :, :], num_goals, axis=1) # (B, N_g, N_d, dim)
        
        # goals: (N_g, dim). Tile across B and N_d
        goals_G_B = np.tile(goals[np.newaxis, :, np.newaxis, :], (batch_size, 1, num_dreams, 1)) # (B, N_g, N_d, dim)
        
        # Swap axes of goals_G_B to match dreams_G_A stacking order: (B, N_d, N_g, dim)
        goals_G_B = np.swapaxes(goals_G_B, 1, 2)
        
        # Flatten to (B * N_d * N_g, dim)
        pairs_a.append(dreams_G_A.reshape(-1, dream_points.shape[-1]))
        pairs_b.append(goals_G_B.reshape(-1, dream_points.shape[-1]))

        
        all_pairs_a = np.concatenate(pairs_a)
        all_pairs_b = np.concatenate(pairs_b)
        
        total_edges = len(all_pairs_a)
        if total_edges > 0:
            # print(f"[CACHE] Full validation: {total_edges} edges (dreams: {num_dreams}, layers: {num_layers})")

            # This is where self._batch_edge_validation is called
            batch_results = self._batch_edge_validation(all_pairs_a, all_pairs_b)
                        
            # 1. Start -> Layer 0
            size_s_1 = batch_size * num_dreams
            free_s_1_flat = batch_results[:size_s_1]
            free_s_1 = free_s_1_flat.reshape(batch_size, num_dreams)
            
            # 2. Layer -> Layer
            offset = size_s_1
            size_layers = batch_size * max(0, num_layers - 1) * num_dreams**2
            
            if num_layers > 1:
                free_layers_flat = batch_results[offset : offset + size_layers]
                free_layers = free_layers_flat.reshape(batch_size, num_layers - 1, num_dreams, num_dreams)
            else:
                free_layers = np.zeros((batch_size, 0, num_dreams, num_dreams), dtype=bool)
                
            # 3. Last Layer -> Goals
            offset += size_layers
            size_final_g = batch_size * num_dreams * num_goals
            free_final_g_flat = batch_results[offset : offset + size_final_g]
            free_final_g = free_final_g_flat.reshape(batch_size, num_dreams, num_goals)
            
        else:
            # No pairs to validate (e.g., num_layers=0 or num_dreams=0, though unlikely)
            free_s_1 = np.zeros((batch_size, num_dreams), dtype=bool)
            free_layers = np.zeros((batch_size, 0, num_dreams, num_dreams), dtype=bool)
            free_final_g = np.zeros((batch_size, num_dreams, num_goals), dtype=bool)

        # Store results in cache
        if cache is not None:
            cache['free_s_1'] = free_s_1.copy()
            cache['free_layers'] = free_layers.copy()
            cache['free_final_g'] = free_final_g.copy()

        return free_s_1, free_layers, free_final_g
    
    def _incremental_validate(
            self,
            q: np.ndarray,
            dream_points: np.ndarray,
            goals: np.ndarray,
            batch_size: int,
            num_layers: int,
            num_dreams: int,
            num_goals: int
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Incremental validation using cached results (vectorized, zero unnecessary copies).

            Key optimizations vs previous version:
            1. expand_cache_if_needed: returns the original array object unchanged when no
                expansion is required — no copy, no allocation.
            2. Subset working arrays are numpy *views* (slices) into the cache arrays, not
                copies.  Because they share memory with cache_X, a single assignment to
                cache_X[b, i] = res is sufficient; the view free_X[b, i] reflects it
                automatically — eliminating the redundant double-write.
            3. The O(edges) Python for-loop over pair_metadata is replaced by three bulk
                numpy advanced-index assignments.  Index arrays are accumulated as plain
                Python lists while building pairs, then converted to numpy once at the end.
            """
            cache = self._cache_container

            prev_n = cache['prev_num_dreams']
            prev_m = cache['prev_num_layers']

            cached_s1     = cache['free_s_1']
            cached_layers = cache['free_layers']
            cached_final_g = cache['free_final_g']

            cache_max_dreams = cached_s1.shape[1]     if cached_s1     is not None else 0
            cache_max_layers = (cached_layers.shape[1] + 1
                                if cached_layers is not None and cached_layers.size > 0
                                else 0)
            cache_max_goals  = cached_final_g.shape[2] if cached_final_g is not None else 0

            max_dreams = max(num_dreams, cache_max_dreams)
            max_layers = max(num_layers, cache_max_layers)
            max_goals  = max(num_goals,  cache_max_goals)

            # ------------------------------------------------------------------
            # 1. Expand cache arrays ONLY when the current shape is insufficient.
            #    Return the *original object* when no reallocation is needed so
            #    that the subsequent view (slice) shares the same memory and no
            #    copy is made.
            # ------------------------------------------------------------------
            def _expand(arr, new_shape):
                """
                Return arr unchanged if it already fits, otherwise allocate a
                zero-filled array of new_shape and copy arr into it.
                No copy is made in the common (no-growth) case.
                """
                if arr is None:
                    return np.zeros(new_shape, dtype=bool)
                if arr.shape == new_shape:
                    return arr                          # ← zero copies, same object
                if all(ns <= os for ns, os in zip(new_shape, arr.shape)):
                    return arr                          # ← fits, no copy needed
                expanded = np.zeros(new_shape, dtype=bool)
                src_slices = tuple(slice(0, s) for s in arr.shape)
                expanded[src_slices] = arr              # one bulk copy, no Python loop
                return expanded

            cache_s1 = _expand(cached_s1, (batch_size, max_dreams))

            if max_layers > 1:
                cache_layers = _expand(
                    cached_layers,
                    (batch_size, max_layers - 1, max_dreams, max_dreams)
                )
            else:
                # Edge case: empty layers tensor — create a fresh zero array (tiny).
                cache_layers = np.zeros((batch_size, 0, max_dreams, max_dreams), dtype=bool)

            cache_final_g = _expand(cached_final_g, (batch_size, max_dreams, max_goals))

            # ------------------------------------------------------------------
            # 2. Working "free_X" arrays are *views* (numpy slices) into cache_X.
            #    Writing cache_X[b, i] = v is immediately visible through free_X
            #    when b < num_X — no separate write to free_X is required.
            #    Note: num_layers == 1 yields an empty view; fall back to a fresh
            #    zero array for the caller's convenience.
            # ------------------------------------------------------------------
            free_s1 = cache_s1[:, :num_dreams]                                          # view

            if num_layers > 1:
                free_layers = cache_layers[:, :num_layers - 1, :num_dreams, :num_dreams]  # view
            else:
                free_layers = np.zeros((batch_size, 0, num_dreams, num_dreams), dtype=bool)

            free_final_g = cache_final_g[:, :num_dreams, :num_goals]                    # view

            # ------------------------------------------------------------------
            # 3. Build validation pairs exactly as before, but replace the
            #    tuple-of-tuples pair_metadata with *typed index lists* so that
            #    we can do one bulk numpy assignment per result type at the end
            #    instead of a Python for-loop over every edge.
            # ------------------------------------------------------------------
            pairs_a: list[np.ndarray] = []
            pairs_b: list[np.ndarray] = []

            # Accumulate flat index arrays per result type.
            s1_b:    list[int] = []; s1_i:    list[int] = []
            lay_b:   list[int] = []; lay_l:   list[int] = []
            lay_i:   list[int] = []; lay_j:   list[int] = []
            fg_b:    list[int] = []; fg_i:    list[int] = []
            fg_j:    list[int] = []

            new_d_slice = slice(prev_n, num_dreams)
            old_d_slice = slice(0, prev_n)
            dim = dream_points.shape[-1]

            # ---- 1. Start → Layer-0 (new dreams only) -------------------------
            if num_dreams > prev_n:
                num_new = num_dreams - prev_n
                pairs_a.append(np.tile(q, (batch_size * num_new, 1)))
                pairs_b.append(dream_points[:, 0, new_d_slice].reshape(-1, dim))

                b_idx = np.repeat(np.arange(batch_size), num_new)
                i_idx = np.tile(np.arange(prev_n, num_dreams), batch_size)
                s1_b.append(b_idx); s1_i.append(i_idx)

            # ---- 2. Layer → Layer connections ---------------------------------
            # Cross-product pair construction: for src-shape (B, Ni, dim) and
            # tgt-shape (B, Nj, dim), we want flat-ordered pairs (a, b) such that
            # at flat index p = b*Ni*Nj + i*Nj + j, a_flat[p] = src[b, i] and
            # b_flat[p] = tgt[b, j].  This requires broadcasting src on axis 2
            # and tgt on axis 1 — inserting newaxis at the wrong slot yields
            # only the diagonal (i == j) and silently misreports collisions.
            def _cross_pairs(src, tgt):
                B_, Ni, D = src.shape
                _,  Nj, _ = tgt.shape
                A = np.broadcast_to(src[:, :, None, :], (B_, Ni, Nj, D))
                Bm = np.broadcast_to(tgt[:, None, :, :], (B_, Ni, Nj, D))
                return (np.ascontiguousarray(A).reshape(-1, D),
                        np.ascontiguousarray(Bm).reshape(-1, D))

            if num_layers > 1:
                if num_layers > prev_m:
                    # New layers: full cross-product for every new layer transition.
                    for l in range(max(0, prev_m - 1), num_layers - 1):
                        dl  = dream_points[:, l]                        # (B, N_d, dim)
                        dl1 = dream_points[:, l + 1]

                        a_flat, b_flat = _cross_pairs(dl, dl1)
                        pairs_a.append(a_flat)
                        pairs_b.append(b_flat)

                        b_idx = np.repeat(np.arange(batch_size), num_dreams * num_dreams)
                        l_idx = np.full_like(b_idx, l)
                        i_idx = np.tile(np.repeat(np.arange(num_dreams), num_dreams), batch_size)
                        j_idx = np.tile(np.tile(np.arange(num_dreams), num_dreams), batch_size)
                        lay_b.append(b_idx); lay_l.append(l_idx)
                        lay_i.append(i_idx); lay_j.append(j_idx)

                elif num_dreams > prev_n:
                    # Existing layers, new dreams: only cross-products involving ≥1 new index.
                    num_new = num_dreams - prev_n
                    for l in range(min(prev_m - 1, num_layers - 1)):
                        dl_old  = dream_points[:, l,     old_d_slice, :]   # (B, N_o, dim)
                        dl_new  = dream_points[:, l,     new_d_slice, :]   # (B, N_n, dim)
                        dl1_old = dream_points[:, l + 1, old_d_slice, :]
                        dl1_new = dream_points[:, l + 1, new_d_slice, :]

                        # Sub-case B1: old → new (i in [0, prev_n), j in [prev_n, num_dreams))
                        a_flat, b_flat = _cross_pairs(dl_old, dl1_new)
                        pairs_a.append(a_flat); pairs_b.append(b_flat)
                        b_idx = np.repeat(np.arange(batch_size), prev_n * num_new)
                        lay_b.append(b_idx); lay_l.append(np.full_like(b_idx, l))
                        lay_i.append(np.tile(np.repeat(np.arange(prev_n), num_new), batch_size))
                        lay_j.append(np.tile(np.tile(np.arange(prev_n, num_dreams), prev_n), batch_size))

                        # Sub-case B2: new → old (i in [prev_n, num_dreams), j in [0, prev_n))
                        a_flat, b_flat = _cross_pairs(dl_new, dl1_old)
                        pairs_a.append(a_flat); pairs_b.append(b_flat)
                        b_idx = np.repeat(np.arange(batch_size), num_new * prev_n)
                        lay_b.append(b_idx); lay_l.append(np.full_like(b_idx, l))
                        lay_i.append(np.tile(np.repeat(np.arange(prev_n, num_dreams), prev_n), batch_size))
                        lay_j.append(np.tile(np.tile(np.arange(prev_n), num_new), batch_size))

                        # Sub-case B3: new → new
                        a_flat, b_flat = _cross_pairs(dl_new, dl1_new)
                        pairs_a.append(a_flat); pairs_b.append(b_flat)
                        b_idx = np.repeat(np.arange(batch_size), num_new * num_new)
                        lay_b.append(b_idx); lay_l.append(np.full_like(b_idx, l))
                        lay_i.append(np.tile(np.repeat(np.arange(prev_n, num_dreams), num_new), batch_size))
                        lay_j.append(np.tile(np.tile(np.arange(prev_n, num_dreams), num_new), batch_size))

            # ---- 3. Last layer → Goals ----------------------------------------
            if num_layers > prev_m:
                # New last layer — revalidate ALL dreams.
                ld = dream_points[:, -1, :]                                      # (B, N_d, dim)
                A = np.repeat(ld[:, np.newaxis, :, :], num_goals, axis=1)
                B = np.swapaxes(
                    np.tile(goals[np.newaxis, :, np.newaxis, :], (batch_size, 1, num_dreams, 1)),
                    1, 2
                )
                pairs_a.append(A.reshape(-1, dim))
                pairs_b.append(B.reshape(-1, dim))
                b_idx = np.repeat(np.arange(batch_size), num_dreams * num_goals)
                fg_b.append(b_idx)
                fg_i.append(np.tile(np.repeat(np.arange(num_dreams), num_goals), batch_size))
                fg_j.append(np.tile(np.tile(np.arange(num_goals), num_dreams), batch_size))

            elif num_dreams > prev_n:
                # Same last layer, new dreams only.
                num_new = num_dreams - prev_n
                ld_new = dream_points[:, -1, new_d_slice, :]                     # (B, N_n, dim)
                A = np.repeat(ld_new[:, np.newaxis, :, :], num_goals, axis=1)
                B = np.swapaxes(
                    np.tile(goals[np.newaxis, :, np.newaxis, :], (batch_size, 1, num_new, 1)),
                    1, 2
                )
                pairs_a.append(A.reshape(-1, dim))
                pairs_b.append(B.reshape(-1, dim))
                b_idx = np.repeat(np.arange(batch_size), num_new * num_goals)
                fg_b.append(b_idx)
                fg_i.append(np.tile(np.repeat(np.arange(prev_n, num_dreams), num_goals), batch_size))
                fg_j.append(np.tile(np.tile(np.arange(num_goals), num_new), batch_size))

            # ------------------------------------------------------------------
            # 4. Run validation and assign results back via bulk numpy indexing.
            #    Three assignments replace the O(edges) Python for-loop.
            #    Because free_X are *views* into cache_X, writing cache_X updates
            #    free_X automatically — no duplicate assignment needed.
            # ------------------------------------------------------------------
            if pairs_a:
                all_a = np.concatenate(pairs_a)
                all_b = np.concatenate(pairs_b)
                results = self._batch_edge_validation(all_a, all_b)

                # Concatenate each typed index list once, then index assign.
                offset = 0

                if s1_b:
                    b = np.concatenate(s1_b); i = np.concatenate(s1_i)
                    n = len(b)
                    cache_s1[b, i] = results[offset: offset + n]
                    offset += n

                if lay_b:
                    b = np.concatenate(lay_b); l = np.concatenate(lay_l)
                    i = np.concatenate(lay_i); j = np.concatenate(lay_j)
                    n = len(b)
                    cache_layers[b, l, i, j] = results[offset: offset + n]
                    offset += n

                if fg_b:
                    b = np.concatenate(fg_b)
                    i = np.concatenate(fg_i); j = np.concatenate(fg_j)
                    n = len(b)
                    cache_final_g[b, i, j] = results[offset: offset + n]

            # ------------------------------------------------------------------
            # 5. Persist expanded arrays back to the cache dict.
            #    When no expansion happened cache_X IS cache['free_X'] already,
            #    so this is a no-op in the common case.
            # ------------------------------------------------------------------
            cache['free_s_1']      = cache_s1
            cache['free_layers']   = cache_layers
            cache['free_final_g']  = cache_final_g
            cache['prev_num_dreams'] = num_dreams
            cache['prev_num_layers'] = num_layers

            return free_s1, free_layers, free_final_g
    @staticmethod
    def pairwise_distance(q1: jax.Array, q2: jax.Array) -> jax.Array:
        """
        Computes pairwise Euclidean distance between two sets of points.
        q1: (..., N, dim)
        q2: (..., M, dim)
        Returns: (..., N, M)
        """
        return jnp.linalg.norm(q1[..., :, None, :] - q2[..., None, :, :], axis=-1)
    

    def _batch_edge_validation(self, pairs_a, pairs_b):
        """
        Batch edge validation using the correct method for self.validation_method.
        Returns a boolean array of results.
        """
        # jax.debug.print("Number of edges to validate: {}", len(pairs_a))
        cost_bound = self._get_cost_bound()
        use_bound = np.isfinite(cost_bound)
        if self.validation_method == 'validate_motion' and hasattr(self.robot_module, 'validate_motion_batch'):
            return self.robot_module.validate_motion_batch(pairs_a, pairs_b, self.env)
        elif self.validation_method == 'rrtc' and hasattr(self.robot_module, 'aorrtc_batch'):
            batch_results = self.robot_module.aorrtc_batch(pairs_a, pairs_b, self.env, self.validation_settings)
            self._update_max_connector_cost(batch_results)
            if use_bound:
                return np.array(
                    [r.solved and (r.path.cost() <= cost_bound) for r in batch_results],
                    dtype=bool,
                )
            return np.array([r.solved for r in batch_results], dtype=bool)
        elif self.validation_method == 'aorrtc' and hasattr(self.robot_module, 'aorrtc_batch'):
            batch_results = self.robot_module.aorrtc_batch(pairs_a, pairs_b, self.env, self.validation_settings)
            self._update_max_connector_cost(batch_results)
            if use_bound:
                return np.array(
                    [r.solved and (r.path.cost() <= cost_bound) for r in batch_results],
                    dtype=bool,
                )
            return np.array([r.solved for r in batch_results], dtype=bool)
        else:
            raise RuntimeError(f"No batch validation available for method '{self.validation_method}'.")
       
    def __call__(self, q: jax.Array, dream_points: jax.Array, goals: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        JAX-compatible cost function that uses VAMP for validation.
        """
        
        batch_size = dream_points.shape[0]
        num_layers = dream_points.shape[1]
        num_dreams = dream_points.shape[2]
        num_goals = goals.shape[0]
        
        # 1. Calculate Distances (Euclidean) - batched
        dist_s_1 = jnp.linalg.norm(dream_points[:, 0] - q[None, None, :], axis=-1)
        
        if num_layers > 1:
            dist_layers = self.pairwise_distance(dream_points[:, :-1], dream_points[:, 1:])
        else:
            dist_layers = jnp.zeros((batch_size, 0, num_dreams, num_dreams))

        dist_final_g = self.pairwise_distance(dream_points[:, -1], goals[None, :])
        
        # 2. Calculate Collisions (VAMP via pure_callback)
        result_shape_s1 = jax.ShapeDtypeStruct((batch_size, num_dreams), bool)
        result_shape_layers = jax.ShapeDtypeStruct((batch_size, num_layers - 1, num_dreams, num_dreams), bool) if num_layers > 1 else jax.ShapeDtypeStruct((batch_size, 0, num_dreams, num_dreams), bool)
        result_shape_final = jax.ShapeDtypeStruct((batch_size, num_dreams, num_goals), bool)
        
        def callback_wrapper(q_val, dp_val, g_val):
            return self._vamp_validate_callback(q_val, dp_val, g_val)

        free_s_1, free_layers, free_final_g = pure_callback(
            callback_wrapper,
            (result_shape_s1, result_shape_layers, result_shape_final),
            q, dream_points, goals,
            vmap_method='sequential'
        )

        # 3. Combine into Costs
        Cs = dist_s_1 + jnp.where(~free_s_1, jnp.inf, 0.0)

        if num_layers > 1:
            Ch = dist_layers + jnp.where(~free_layers, jnp.inf, 0.0)
        else:
            Ch = dist_layers

        Cl = dist_final_g + jnp.where(~free_final_g, jnp.inf, 0.0)

        Cg = -jnp.ones(num_goals)

        return Cs, Ch, Cl, Cg


