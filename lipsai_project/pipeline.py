from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import expit


@dataclass
class LPCache:
    A_ub: csr_matrix
    b_ub: np.ndarray
    bounds: List[Tuple[float, float]]
    lower: np.ndarray
    upper: np.ndarray
    n: int


@dataclass
class LPGeometry:
    """Pairwise geometry shared by the full LP and cutting-plane solver."""

    x: np.ndarray
    y: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    bounds: List[Tuple[float, float]]
    iu: np.ndarray
    ju: np.ndarray
    d_same: np.ndarray
    d_flip: np.ndarray
    n: int

    @property
    def n_pairs(self) -> int:
        return int(self.iu.size)

    @property
    def n_candidate_constraints(self) -> int:
        return int(4 * self.iu.size)


@dataclass
class SolveMetrics:
    algorithm: str
    value: float
    n_constraints: int
    iterations: int
    converged: bool
    max_violation: float
    lp_solve_sec: float
    separation_sec: float
    matrix_build_sec: float
    active_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


@dataclass
class TestResult:
    statistic: float
    critical_value: float
    p_value: float
    reject: bool
    mc_values: np.ndarray
    solution: np.ndarray
    runtime_sec: float
    algorithm: str = 'efficient'
    n_full_constraints: int = 0
    obs_constraints: int = 0
    mean_mc_constraints: float = 0.0
    obs_iterations: int = 0
    mean_mc_iterations: float = 0.0
    max_violation: float = 0.0
    geometry_build_sec: float = 0.0
    matrix_build_sec: float = 0.0
    lp_solve_sec: float = 0.0
    separation_sec: float = 0.0


# ---------------------------------------------------------------------------
# Data transforms and finite-dimensional LP geometry
# ---------------------------------------------------------------------------


def sigmoid_transform(arr: np.ndarray) -> np.ndarray:
    return expit(arr)


def build_lipsai_geometry(x: np.ndarray, y: np.ndarray) -> LPGeometry:
    """Build the shared O(n^2) pairwise distances once.

    The variable v_i represents u(Y_i, X_i). The sign bounds encode the
    antisymmetry/SAI restriction at each observed pair. Pair constraints encode
    the Lipschitz and antisymmetry-coupled Lipschitz inequalities.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape.')
    if x.ndim != 1:
        raise ValueError('x and y must be one-dimensional arrays.')

    n = x.size
    diag_bound = np.abs(x - y) / math.sqrt(2.0)
    tol = 1e-12
    lower = np.empty(n, dtype=float)
    upper = np.empty(n, dtype=float)
    bounds: List[Tuple[float, float]] = []

    for i, (xi, yi, bi) in enumerate(zip(x, y, diag_bound)):
        bi = float(bi)
        if xi > yi + tol:
            li, ui = 0.0, bi
        elif yi > xi + tol:
            li, ui = -bi, 0.0
        else:
            li, ui = 0.0, 0.0
        lower[i] = li
        upper[i] = ui
        bounds.append((li, ui))

    iu, ju = np.triu_indices(n, k=1)
    if iu.size == 0:
        d_same = np.empty(0, dtype=float)
        d_flip = np.empty(0, dtype=float)
    else:
        d_same = np.sqrt((x[iu] - x[ju]) ** 2 + (y[iu] - y[ju]) ** 2)
        d_flip = np.sqrt((x[iu] - y[ju]) ** 2 + (y[iu] - x[ju]) ** 2)

    return LPGeometry(
        x=x,
        y=y,
        lower=lower,
        upper=upper,
        bounds=bounds,
        iu=iu,
        ju=ju,
        d_same=d_same,
        d_flip=d_flip,
        n=n,
    )


def _append_two_var_rows(
    row_ids: np.ndarray,
    col_a: np.ndarray,
    col_b: np.ndarray,
    coef_a: np.ndarray,
    coef_b: np.ndarray,
    rhs: np.ndarray,
    row_chunks: List[np.ndarray],
    col_chunks: List[np.ndarray],
    data_chunks: List[np.ndarray],
    rhs_chunks: List[np.ndarray],
) -> None:
    if row_ids.size == 0:
        return
    row_chunks.extend([row_ids, row_ids])
    col_chunks.extend([col_a, col_b])
    data_chunks.extend([coef_a, coef_b])
    rhs_chunks.append(rhs)


def _constraint_count(geo: LPGeometry, *, prune_redundant: bool = True) -> int:
    m = geo.n_pairs
    if m == 0:
        return 0
    if not prune_redundant:
        return int(4 * m)

    iu = geo.iu
    ju = geo.ju
    lower = geo.lower
    upper = geo.upper
    d_same = geo.d_same
    d_flip = geo.d_flip
    tol = 1e-12
    return int(
        np.sum((upper[iu] - lower[ju]) > (d_same + tol))
        + np.sum((upper[ju] - lower[iu]) > (d_same + tol))
        + np.sum((upper[iu] + upper[ju]) > (d_flip + tol))
        + np.sum((-lower[iu] - lower[ju]) > (d_flip + tol))
    )


def build_lipsai_lp_from_geometry(
    geo: LPGeometry,
    *,
    prune_redundant: bool = True,
) -> LPCache:
    """Build the old/full LP constraint matrix in sparse CSR format."""

    n = geo.n
    m = geo.n_pairs
    if m == 0:
        return LPCache(
            A_ub=coo_matrix((0, n)).tocsr(),
            b_ub=np.empty(0, dtype=float),
            bounds=geo.bounds,
            lower=geo.lower,
            upper=geo.upper,
            n=n,
        )

    iu = geo.iu
    ju = geo.ju
    d_same = geo.d_same
    d_flip = geo.d_flip
    lower = geo.lower
    upper = geo.upper

    row_chunks: List[np.ndarray] = []
    col_chunks: List[np.ndarray] = []
    data_chunks: List[np.ndarray] = []
    rhs_chunks: List[np.ndarray] = []
    row_start = 0
    tol = 1e-12

    mask = np.ones(m, dtype=bool)
    if prune_redundant:
        mask = (upper[iu] - lower[ju]) > (d_same + tol)
    if np.any(mask):
        idx = np.flatnonzero(mask)
        rows = row_start + np.arange(idx.size)
        _append_two_var_rows(
            rows,
            iu[idx],
            ju[idx],
            np.ones(idx.size),
            -np.ones(idx.size),
            d_same[idx],
            row_chunks,
            col_chunks,
            data_chunks,
            rhs_chunks,
        )
        row_start += idx.size

    mask = np.ones(m, dtype=bool)
    if prune_redundant:
        mask = (upper[ju] - lower[iu]) > (d_same + tol)
    if np.any(mask):
        idx = np.flatnonzero(mask)
        rows = row_start + np.arange(idx.size)
        _append_two_var_rows(
            rows,
            iu[idx],
            ju[idx],
            -np.ones(idx.size),
            np.ones(idx.size),
            d_same[idx],
            row_chunks,
            col_chunks,
            data_chunks,
            rhs_chunks,
        )
        row_start += idx.size

    mask = np.ones(m, dtype=bool)
    if prune_redundant:
        mask = (upper[iu] + upper[ju]) > (d_flip + tol)
    if np.any(mask):
        idx = np.flatnonzero(mask)
        rows = row_start + np.arange(idx.size)
        _append_two_var_rows(
            rows,
            iu[idx],
            ju[idx],
            np.ones(idx.size),
            np.ones(idx.size),
            d_flip[idx],
            row_chunks,
            col_chunks,
            data_chunks,
            rhs_chunks,
        )
        row_start += idx.size

    mask = np.ones(m, dtype=bool)
    if prune_redundant:
        mask = (-lower[iu] - lower[ju]) > (d_flip + tol)
    if np.any(mask):
        idx = np.flatnonzero(mask)
        rows = row_start + np.arange(idx.size)
        _append_two_var_rows(
            rows,
            iu[idx],
            ju[idx],
            -np.ones(idx.size),
            -np.ones(idx.size),
            d_flip[idx],
            row_chunks,
            col_chunks,
            data_chunks,
            rhs_chunks,
        )
        row_start += idx.size

    if rhs_chunks:
        A_ub = coo_matrix(
            (
                np.concatenate(data_chunks),
                (np.concatenate(row_chunks), np.concatenate(col_chunks)),
            ),
            shape=(row_start, n),
        ).tocsr()
        b_ub = np.concatenate(rhs_chunks)
    else:
        A_ub = coo_matrix((0, n)).tocsr()
        b_ub = np.empty(0, dtype=float)

    return LPCache(
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=geo.bounds,
        lower=geo.lower,
        upper=geo.upper,
        n=n,
    )


def build_lipsai_lp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prune_redundant: bool = True,
) -> LPCache:
    """Backward-compatible full sparse LP builder."""

    geo = build_lipsai_geometry(x, y)
    return build_lipsai_lp_from_geometry(geo, prune_redundant=prune_redundant)


# ---------------------------------------------------------------------------
# LP solvers: old full method and new cutting-plane method
# ---------------------------------------------------------------------------


def solve_lp(
    weights: np.ndarray,
    cache: LPCache,
    *,
    method: str = 'highs-ds',
    presolve: bool = False,
    return_solution: bool = False,
) -> Tuple[float, Optional[np.ndarray]]:
    weights = np.asarray(weights, dtype=float)
    res = linprog(
        c=-weights,
        A_ub=cache.A_ub,
        b_ub=cache.b_ub,
        bounds=cache.bounds,
        method=method,
        options={'presolve': presolve},
    )
    if not res.success:
        raise RuntimeError(f'LP solver failed: {res.message}')
    value = float(weights @ res.x)
    solution = np.asarray(res.x, dtype=float) if return_solution else None
    return value, solution


def _active_lp_from_ids(geo: LPGeometry, active_ids: np.ndarray) -> LPCache:
    """Build sparse LP matrix for encoded active constraints.

    Encoding: type * n_pairs + pair_index, where the four types are
      0:  v_i - v_j <= d_same
      1: -v_i + v_j <= d_same
      2:  v_i + v_j <= d_flip
      3: -v_i - v_j <= d_flip
    """

    active_ids = np.asarray(active_ids, dtype=np.int64)
    k = int(active_ids.size)
    if k == 0:
        return LPCache(
            A_ub=coo_matrix((0, geo.n)).tocsr(),
            b_ub=np.empty(0, dtype=float),
            bounds=geo.bounds,
            lower=geo.lower,
            upper=geo.upper,
            n=geo.n,
        )

    m = geo.n_pairs
    ctype = active_ids // m
    pair_idx = active_ids % m
    i = geo.iu[pair_idx]
    j = geo.ju[pair_idx]

    coef_i = np.empty(k, dtype=float)
    coef_j = np.empty(k, dtype=float)
    rhs = np.empty(k, dtype=float)

    mask = ctype == 0
    coef_i[mask] = 1.0
    coef_j[mask] = -1.0
    rhs[mask] = geo.d_same[pair_idx[mask]]

    mask = ctype == 1
    coef_i[mask] = -1.0
    coef_j[mask] = 1.0
    rhs[mask] = geo.d_same[pair_idx[mask]]

    mask = ctype == 2
    coef_i[mask] = 1.0
    coef_j[mask] = 1.0
    rhs[mask] = geo.d_flip[pair_idx[mask]]

    mask = ctype == 3
    coef_i[mask] = -1.0
    coef_j[mask] = -1.0
    rhs[mask] = geo.d_flip[pair_idx[mask]]

    rows = np.arange(k, dtype=np.int64)
    A_ub = coo_matrix(
        (
            np.concatenate([coef_i, coef_j]),
            (np.concatenate([rows, rows]), np.concatenate([i, j])),
        ),
        shape=(k, geo.n),
    ).tocsr()

    return LPCache(
        A_ub=A_ub,
        b_ub=rhs,
        bounds=geo.bounds,
        lower=geo.lower,
        upper=geo.upper,
        n=geo.n,
    )


def _all_constraint_violations(solution: np.ndarray, geo: LPGeometry) -> np.ndarray:
    m = geo.n_pairs
    if m == 0:
        return np.empty(0, dtype=float)

    vi = solution[geo.iu]
    vj = solution[geo.ju]
    violations = np.empty(4 * m, dtype=float)
    violations[0:m] = vi - vj - geo.d_same
    violations[m : 2 * m] = -vi + vj - geo.d_same
    violations[2 * m : 3 * m] = vi + vj - geo.d_flip
    violations[3 * m : 4 * m] = -vi - vj - geo.d_flip
    return violations


def _select_violated_constraints(
    solution: np.ndarray,
    geo: LPGeometry,
    *,
    active_mask: np.ndarray,
    violation_tol: float,
    max_add: int,
) -> Tuple[np.ndarray, float]:
    violations = _all_constraint_violations(solution, geo)
    if violations.size == 0:
        return np.empty(0, dtype=np.int64), 0.0

    max_violation = float(np.max(violations))
    violations[active_mask] = -np.inf
    candidates = np.flatnonzero(violations > violation_tol)
    if candidates.size == 0:
        return np.empty(0, dtype=np.int64), max_violation

    if max_add > 0 and candidates.size > max_add:
        # Add the largest violations only; this keeps each LP subproblem moderate.
        local = np.argpartition(violations[candidates], -max_add)[-max_add:]
        candidates = candidates[local]

    # Stable order makes repeated runs easier to debug.
    candidates = candidates[np.argsort(candidates)]
    return candidates.astype(np.int64, copy=False), max_violation


def solve_lp_cutting_plane(
    weights: np.ndarray,
    geo: LPGeometry,
    *,
    method: str = 'highs-ds',
    presolve: bool = False,
    violation_tol: float = 1e-8,
    max_iter: int = 50,
    max_add: int = 5000,
    initial_active_ids: Optional[np.ndarray] = None,
    return_solution: bool = False,
    verbose: bool = False,
) -> Tuple[float, Optional[np.ndarray], SolveMetrics]:
    """Solve the Lip-SAI finite LP by delayed constraint generation.

    The method is exact up to `violation_tol`: after each restricted LP solve we
    check all omitted Lipschitz/antisymmetry constraints, add the most violated
    ones, and repeat until no violation remains.
    """

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (geo.n,):
        raise ValueError(f'weights must have shape ({geo.n},).')

    total_constraints = geo.n_candidate_constraints
    active_mask = np.zeros(total_constraints, dtype=bool)
    if initial_active_ids is None or len(initial_active_ids) == 0:
        active_ids = np.empty(0, dtype=np.int64)
    else:
        active_ids = np.unique(np.asarray(initial_active_ids, dtype=np.int64))
        active_ids = active_ids[(0 <= active_ids) & (active_ids < total_constraints)]
        active_mask[active_ids] = True

    lp_solve_sec = 0.0
    separation_sec = 0.0
    matrix_build_sec = 0.0
    solution = None
    value = np.nan
    max_violation = np.inf
    converged = False
    iterations_done = 0

    for iteration in range(max_iter + 1):
        t_build = time.perf_counter()
        cache = _active_lp_from_ids(geo, active_ids)
        matrix_build_sec += time.perf_counter() - t_build

        t_solve = time.perf_counter()
        value, solution = solve_lp(
            weights,
            cache,
            method=method,
            presolve=presolve,
            return_solution=True,
        )
        lp_solve_sec += time.perf_counter() - t_solve

        t_sep = time.perf_counter()
        new_ids, max_violation = _select_violated_constraints(
            solution,
            geo,
            active_mask=active_mask,
            violation_tol=violation_tol,
            max_add=max_add,
        )
        separation_sec += time.perf_counter() - t_sep
        iterations_done = iteration + 1

        if verbose:
            print(
                f'iter={iteration:02d} | value={value:.10f} | '
                f'constraints={active_ids.size} | max_viol={max_violation:.3e} | '
                f'add={new_ids.size}'
            )

        if max_violation <= violation_tol or new_ids.size == 0:
            converged = True
            break

        active_mask[new_ids] = True
        active_ids = np.concatenate([active_ids, new_ids])

    if solution is None:
        raise RuntimeError('Cutting-plane solver failed before producing a solution.')
    if not converged:
        raise RuntimeError(
            'Cutting-plane solver did not converge. '
            f'max_violation={max_violation:.3e}, active_constraints={active_ids.size}. '
            'Increase --cutting-plane-max-iter or --cutting-plane-max-add.'
        )

    metrics = SolveMetrics(
        algorithm='efficient',
        value=float(value),
        n_constraints=int(active_ids.size),
        iterations=int(iterations_done),
        converged=bool(converged),
        max_violation=float(max(0.0, max_violation)),
        lp_solve_sec=float(lp_solve_sec),
        separation_sec=float(separation_sec),
        matrix_build_sec=float(matrix_build_sec),
        active_ids=active_ids.copy(),
    )
    return float(value), (solution if return_solution else None), metrics


def solve_many_lp(
    weights_mat: np.ndarray,
    cache: LPCache,
    *,
    method: str = 'highs-ds',
    presolve: bool = False,
) -> np.ndarray:
    weights_mat = np.asarray(weights_mat, dtype=float)
    if weights_mat.ndim != 2:
        raise ValueError('weights_mat must be 2D')
    values = []
    for weights in weights_mat:
        value, _ = solve_lp(weights, cache, method=method, presolve=presolve, return_solution=False)
        values.append(value)
    return np.asarray(values, dtype=float)


# ---------------------------------------------------------------------------
# Monte Carlo test pipeline
# ---------------------------------------------------------------------------


def monte_carlo_lipsai_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.05,
    B: int = 199,
    rng: np.random.Generator | None = None,
    method: str = 'highs-ds',
    presolve: bool = False,
    prune_redundant: bool = True,
    apply_sigmoid_transform: bool = True,
    algorithm: str = 'efficient',
    cutting_plane_tol: float = 1e-8,
    cutting_plane_max_iter: int = 50,
    cutting_plane_max_add: int = 5000,
    cutting_plane_warm_start: bool = True,
    cutting_plane_verbose: bool = False,
) -> TestResult:
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if apply_sigmoid_transform:
        xy = sigmoid_transform(np.column_stack([x, y]))
        x = xy[:, 0]
        y = xy[:, 1]

    if algorithm not in {'efficient', 'full'}:
        raise ValueError("algorithm must be either 'efficient' or 'full'.")

    t_total = time.perf_counter()
    t_geo = time.perf_counter()
    geo = build_lipsai_geometry(x, y)
    geometry_build_sec = time.perf_counter() - t_geo
    n = x.size
    n_full_constraints = _constraint_count(geo, prune_redundant=prune_redundant)

    obs_weights = np.full(n, 2.0 / n, dtype=float)
    xi = rng.integers(0, 2, size=(B, n), dtype=np.int8)
    xi = 2.0 * xi.astype(float) - 1.0
    weights_mat = (2.0 / n) * xi

    if algorithm == 'full':
        t_build = time.perf_counter()
        cache = build_lipsai_lp_from_geometry(geo, prune_redundant=prune_redundant)
        matrix_build_sec = time.perf_counter() - t_build

        t_solve = time.perf_counter()
        statistic, solution = solve_lp(
            obs_weights,
            cache,
            method=method,
            presolve=presolve,
            return_solution=True,
        )
        mc_values = solve_many_lp(
            weights_mat,
            cache,
            method=method,
            presolve=presolve,
        )
        lp_solve_sec = time.perf_counter() - t_solve
        separation_sec = 0.0
        obs_constraints = int(cache.A_ub.shape[0])
        mean_mc_constraints = float(cache.A_ub.shape[0])
        obs_iterations = 1
        mean_mc_iterations = 1.0
        max_violation = 0.0
    else:
        statistic, solution, obs_metrics = solve_lp_cutting_plane(
            obs_weights,
            geo,
            method=method,
            presolve=presolve,
            violation_tol=cutting_plane_tol,
            max_iter=cutting_plane_max_iter,
            max_add=cutting_plane_max_add,
            initial_active_ids=None,
            return_solution=True,
            verbose=cutting_plane_verbose,
        )

        mc_values = []
        mc_constraints = []
        mc_iterations = []
        mc_max_violations = []
        total_lp_solve_sec = obs_metrics.lp_solve_sec
        total_separation_sec = obs_metrics.separation_sec
        total_matrix_build_sec = obs_metrics.matrix_build_sec
        warm_ids = obs_metrics.active_ids if cutting_plane_warm_start else None

        for weights in weights_mat:
            value, _, metrics = solve_lp_cutting_plane(
                weights,
                geo,
                method=method,
                presolve=presolve,
                violation_tol=cutting_plane_tol,
                max_iter=cutting_plane_max_iter,
                max_add=cutting_plane_max_add,
                initial_active_ids=warm_ids,
                return_solution=False,
                verbose=False,
            )
            mc_values.append(value)
            mc_constraints.append(metrics.n_constraints)
            mc_iterations.append(metrics.iterations)
            mc_max_violations.append(metrics.max_violation)
            total_lp_solve_sec += metrics.lp_solve_sec
            total_separation_sec += metrics.separation_sec
            total_matrix_build_sec += metrics.matrix_build_sec

        mc_values = np.asarray(mc_values, dtype=float)
        matrix_build_sec = total_matrix_build_sec
        lp_solve_sec = total_lp_solve_sec
        separation_sec = total_separation_sec
        obs_constraints = int(obs_metrics.n_constraints)
        mean_mc_constraints = float(np.mean(mc_constraints)) if mc_constraints else 0.0
        obs_iterations = int(obs_metrics.iterations)
        mean_mc_iterations = float(np.mean(mc_iterations)) if mc_iterations else 0.0
        max_violation = float(max([obs_metrics.max_violation] + mc_max_violations)) if mc_max_violations else obs_metrics.max_violation

    runtime_sec = time.perf_counter() - t_total
    ge_count = int(np.sum(mc_values >= statistic))
    p_value = (1.0 + ge_count) / (B + 1.0)
    critical_value = float(np.quantile(mc_values, 1.0 - alpha, method='higher'))
    reject = bool(p_value <= alpha)

    return TestResult(
        statistic=float(statistic),
        critical_value=critical_value,
        p_value=float(p_value),
        reject=reject,
        mc_values=mc_values,
        solution=solution if solution is not None else np.empty(n, dtype=float),
        runtime_sec=float(runtime_sec),
        algorithm=algorithm,
        n_full_constraints=int(n_full_constraints),
        obs_constraints=int(obs_constraints),
        mean_mc_constraints=float(mean_mc_constraints),
        obs_iterations=int(obs_iterations),
        mean_mc_iterations=float(mean_mc_iterations),
        max_violation=float(max_violation),
        geometry_build_sec=float(geometry_build_sec),
        matrix_build_sec=float(matrix_build_sec),
        lp_solve_sec=float(lp_solve_sec),
        separation_sec=float(separation_sec),
    )


def _summary_stat(detail: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in detail.columns or detail[column].empty:
        return default
    return float(detail[column].mean())


def run_repeated_test(
    x_sampler,
    *,
    n: int,
    n_rep: int,
    alpha: float,
    B: int,
    seed: int,
    apply_sigmoid_transform: bool = True,
    progress_every: int = 10,
    method: str = 'highs-ds',
    presolve: bool = False,
    prune_redundant: bool = True,
    algorithm: str = 'efficient',
    cutting_plane_tol: float = 1e-8,
    cutting_plane_max_iter: int = 50,
    cutting_plane_max_add: int = 5000,
    cutting_plane_warm_start: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []

    for rep in range(1, n_rep + 1):
        x, y = x_sampler(n=n, rng=rng)
        out = monte_carlo_lipsai_test(
            x,
            y,
            alpha=alpha,
            B=B,
            rng=rng,
            method=method,
            presolve=presolve,
            prune_redundant=prune_redundant,
            apply_sigmoid_transform=apply_sigmoid_transform,
            algorithm=algorithm,
            cutting_plane_tol=cutting_plane_tol,
            cutting_plane_max_iter=cutting_plane_max_iter,
            cutting_plane_max_add=cutting_plane_max_add,
            cutting_plane_warm_start=cutting_plane_warm_start,
        )
        rows.append(
            {
                'rep': rep,
                'n': n,
                'alpha': alpha,
                'B': B,
                'statistic': out.statistic,
                'critical_value': out.critical_value,
                'p_value': out.p_value,
                'reject': int(out.reject),
                'runtime_sec': out.runtime_sec,
                'geometry_build_sec': out.geometry_build_sec,
                'matrix_build_sec': out.matrix_build_sec,
                'lp_solve_sec': out.lp_solve_sec,
                'separation_sec': out.separation_sec,
                'n_full_constraints': out.n_full_constraints,
                'obs_constraints': out.obs_constraints,
                'mean_mc_constraints': out.mean_mc_constraints,
                'obs_iterations': out.obs_iterations,
                'mean_mc_iterations': out.mean_mc_iterations,
                'max_violation': out.max_violation,
                'apply_sigmoid_transform': int(apply_sigmoid_transform),
                'algorithm': out.algorithm,
                'method': method,
                'presolve': int(presolve),
                'prune_redundant': int(prune_redundant),
                'cutting_plane_tol': cutting_plane_tol,
                'cutting_plane_max_iter': cutting_plane_max_iter,
                'cutting_plane_max_add': cutting_plane_max_add,
                'cutting_plane_warm_start': int(cutting_plane_warm_start),
            }
        )
        if progress_every > 0 and (rep % progress_every == 0 or rep == n_rep):
            partial = pd.DataFrame(rows)
            print(
                f'rep={rep:3d}/{n_rep}  '
                f'reject_rate={partial["reject"].mean():.3f}  '
                f'mean_p={partial["p_value"].mean():.4f}  '
                f'mean_time={partial["runtime_sec"].mean():.3f}s  '
                f'mean_constraints={partial["obs_constraints"].mean():.0f}'
            )

    detail = pd.DataFrame(rows)
    summary: Dict[str, float] = {
        'n': float(n),
        'n_rep': float(n_rep),
        'alpha': float(alpha),
        'B': float(B),
        'reject_rate': float(detail['reject'].mean()),
        'mean_p_value': float(detail['p_value'].mean()),
        'median_p_value': float(detail['p_value'].median()),
        'mean_statistic': float(detail['statistic'].mean()),
        'sd_statistic': float(detail['statistic'].std(ddof=1) if len(detail) > 1 else 0.0),
        'mean_critical_value': float(detail['critical_value'].mean()),
        'mean_runtime_sec': float(detail['runtime_sec'].mean()),
        'median_runtime_sec': float(detail['runtime_sec'].median()),
        'mean_geometry_build_sec': _summary_stat(detail, 'geometry_build_sec'),
        'mean_matrix_build_sec': _summary_stat(detail, 'matrix_build_sec'),
        'mean_lp_solve_sec': _summary_stat(detail, 'lp_solve_sec'),
        'mean_separation_sec': _summary_stat(detail, 'separation_sec'),
        'mean_n_full_constraints': _summary_stat(detail, 'n_full_constraints'),
        'mean_obs_constraints': _summary_stat(detail, 'obs_constraints'),
        'mean_mc_constraints': _summary_stat(detail, 'mean_mc_constraints'),
        'mean_obs_iterations': _summary_stat(detail, 'obs_iterations'),
        'mean_mc_iterations': _summary_stat(detail, 'mean_mc_iterations'),
        'max_violation': float(detail['max_violation'].max()),
        'algorithm': algorithm,
        'method': method,
        'presolve': float(presolve),
        'prune_redundant': float(prune_redundant),
        'apply_sigmoid_transform': float(apply_sigmoid_transform),
    }
    return detail, summary
