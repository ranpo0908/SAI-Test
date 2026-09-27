from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PACKAGE_DIR = PROJECT_ROOT / 'lipsai_project'
if not PACKAGE_DIR.exists():
    raise ModuleNotFoundError(
        "Could not find the 'lipsai_project' package folder next to run_lipsai_project.py. "
        'Make sure you unzip the full project bundle and run the script from that folder.'
    )

import numpy as np

from lipsai_project.distributions import (
    CASE_REGISTRY,
    ExperimentCase,
    list_case_names,
    make_skew_power_case,
)


DEFAULT_CASES = ['null_l_sai', 'alt_mean_cov', 'alt_var', 'skew_14', 'skew_41']
DEFAULT_N_LIST = [50, 100, 200]
DEFAULT_B_LIST = [99, 199]



def _resolve_cases(case_names: Iterable[str]) -> List[ExperimentCase]:
    cases = []
    for case_name in case_names:
        if case_name not in CASE_REGISTRY:
            raise ValueError(f'Unknown case: {case_name}. Available cases: {", ".join(list_case_names())}')
        cases.append(CASE_REGISTRY[case_name])
    return cases



def _expanded_algorithms(algorithm: str) -> list[str]:
    if algorithm == 'both':
        return ['efficient', 'full']
    return [algorithm]



def _run_case_grid(
    case: ExperimentCase,
    *,
    n_list: Iterable[int],
    B_list: Iterable[int],
    n_rep: int,
    alpha: float,
    seed: int,
    progress_every: int,
    method: str,
    presolve: bool,
    apply_sigmoid_transform: bool,
    prune_redundant: bool,
    algorithm: str,
    cutting_plane_tol: float,
    cutting_plane_max_iter: int,
    cutting_plane_max_add: int,
    cutting_plane_warm_start: bool,
    extra_summary: dict[str, object] | None = None,
    extra_detail: dict[str, object] | None = None,
) -> tuple[object, object]:
    import pandas as pd
    from lipsai_project.pipeline import run_repeated_test
    from lipsai_project.reporting import format_summary_block

    detail_frames = []
    summary_rows = []

    n_list = list(n_list)
    B_list = list(B_list)
    child_seed = seed

    for algo in _expanded_algorithms(algorithm):
        for n in n_list:
            for B in B_list:
                print('=' * 72)
                print(f'Case={case.name}, algorithm={algo}, n={n}, B={B}')
                detail_df, summary = run_repeated_test(
                    case.sampler,
                    n=n,
                    n_rep=n_rep,
                    alpha=alpha,
                    B=B,
                    seed=child_seed,
                    apply_sigmoid_transform=apply_sigmoid_transform,
                    progress_every=progress_every,
                    method=method,
                    presolve=presolve,
                    prune_redundant=prune_redundant,
                    algorithm=algo,
                    cutting_plane_tol=cutting_plane_tol,
                    cutting_plane_max_iter=cutting_plane_max_iter,
                    cutting_plane_max_add=cutting_plane_max_add,
                    cutting_plane_warm_start=cutting_plane_warm_start,
                )
                child_seed += 10_000

                detail_df['case'] = case.name
                detail_df['family'] = case.family
                detail_df['truth'] = case.truth
                detail_df['description'] = case.description
                if extra_detail:
                    for key, value in extra_detail.items():
                        detail_df[key] = value
                detail_frames.append(detail_df)

                summary['case'] = case.name
                summary['family'] = case.family
                summary['truth'] = case.truth
                summary['description'] = case.description
                if extra_summary:
                    summary.update(extra_summary)
                summary_rows.append(summary)
                print(format_summary_block(summary))

    return pd.concat(detail_frames, ignore_index=True), pd.DataFrame(summary_rows)



def run_suite(
    *,
    cases: Iterable[str],
    n_list: Iterable[int],
    B_list: Iterable[int],
    n_rep: int,
    alpha: float,
    seed: int,
    outdir: Path,
    progress_every: int,
    method: str,
    presolve: bool,
    apply_sigmoid_transform: bool,
    prune_redundant: bool,
    algorithm: str,
    cutting_plane_tol: float,
    cutting_plane_max_iter: int,
    cutting_plane_max_add: int,
    cutting_plane_warm_start: bool,
) -> tuple[object, object]:
    import pandas as pd
    from lipsai_project.reporting import save_experiment_outputs

    case_objects = _resolve_cases(cases)
    all_detail = []
    all_summary = []

    for idx, case in enumerate(case_objects):
        detail_df, summary_df = _run_case_grid(
            case,
            n_list=n_list,
            B_list=B_list,
            n_rep=n_rep,
            alpha=alpha,
            seed=seed + idx * 100_000,
            progress_every=progress_every,
            method=method,
            presolve=presolve,
            apply_sigmoid_transform=apply_sigmoid_transform,
            prune_redundant=prune_redundant,
            algorithm=algorithm,
            cutting_plane_tol=cutting_plane_tol,
            cutting_plane_max_iter=cutting_plane_max_iter,
            cutting_plane_max_add=cutting_plane_max_add,
            cutting_plane_warm_start=cutting_plane_warm_start,
        )
        case_dir = outdir / case.name
        save_experiment_outputs(
            outdir=case_dir,
            detail_df=detail_df,
            summary_df=summary_df,
            basename=case.name,
        )
        all_detail.append(detail_df)
        all_summary.append(summary_df)

    combined_detail = pd.concat(all_detail, ignore_index=True)
    combined_summary = pd.concat(all_summary, ignore_index=True)
    save_experiment_outputs(
        outdir=outdir,
        detail_df=combined_detail,
        summary_df=combined_summary,
        basename='all_cases',
    )
    return combined_detail, combined_summary



def _make_a_grid(a_list: Sequence[float] | None, a_min: float, a_max: float, a_step: float) -> list[float]:
    if a_list is not None and len(a_list) > 0:
        return [float(a) for a in a_list]
    if a_step <= 0:
        raise ValueError('--a-step must be positive.')
    grid = np.arange(a_min, a_max + 0.5 * a_step, a_step, dtype=float)
    return [float(a) for a in grid]



def run_skew_power_sweep(
    *,
    a_values: Sequence[float],
    n_list: Iterable[int],
    B_list: Iterable[int],
    n_rep: int,
    alpha: float,
    seed: int,
    outdir: Path,
    progress_every: int,
    method: str,
    presolve: bool,
    apply_sigmoid_transform: bool,
    prune_redundant: bool,
    algorithm: str,
    cutting_plane_tol: float,
    cutting_plane_max_iter: int,
    cutting_plane_max_add: int,
    cutting_plane_warm_start: bool,
) -> tuple[object, object]:
    import pandas as pd
    from lipsai_project.reporting import save_experiment_outputs

    all_detail = []
    all_summary = []
    sweep_dir = outdir / 'skew_power_sweep'

    for idx, a in enumerate(a_values):
        case = make_skew_power_case(float(a))
        detail_df, summary_df = _run_case_grid(
            case,
            n_list=n_list,
            B_list=B_list,
            n_rep=n_rep,
            alpha=alpha,
            seed=seed + idx * 100_000,
            progress_every=progress_every,
            method=method,
            presolve=presolve,
            apply_sigmoid_transform=apply_sigmoid_transform,
            prune_redundant=prune_redundant,
            algorithm=algorithm,
            cutting_plane_tol=cutting_plane_tol,
            cutting_plane_max_iter=cutting_plane_max_iter,
            cutting_plane_max_add=cutting_plane_max_add,
            cutting_plane_warm_start=cutting_plane_warm_start,
            extra_summary={'a': float(a)},
            extra_detail={'a': float(a)},
        )
        case_dir = sweep_dir / case.name
        save_experiment_outputs(
            outdir=case_dir,
            detail_df=detail_df,
            summary_df=summary_df,
            basename=case.name,
        )
        all_detail.append(detail_df)
        all_summary.append(summary_df)

    combined_detail = pd.concat(all_detail, ignore_index=True)
    combined_summary = pd.concat(all_summary, ignore_index=True)
    save_experiment_outputs(
        outdir=sweep_dir,
        detail_df=combined_detail,
        summary_df=combined_summary,
        basename='skew_power_sweep',
    )
    return combined_detail, combined_summary



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Lip-SAI experiment runner with full LP and efficient cutting-plane algorithms.'
    )
    parser.add_argument(
        '--mode',
        choices=['list-cases', 'run-case', 'run-suite', 'run-skew-power'],
        default='run-suite',
        help=(
            'list-cases: print available distributions; run-case: run selected cases; '
            'run-suite: run the default suite; run-skew-power: sweep a in '
            'SN([0,a], [[1,0.3],[0.3,4]], [1,4]).'
        ),
    )
    parser.add_argument('--cases', nargs='+', default=DEFAULT_CASES)
    parser.add_argument('--n-list', nargs='+', type=int, default=DEFAULT_N_LIST)
    parser.add_argument('--B-list', nargs='+', type=int, default=DEFAULT_B_LIST)
    parser.add_argument('--n-rep', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=20260416)
    parser.add_argument('--outdir', type=str, default='lipsai_project_results')
    parser.add_argument('--progress-every', type=int, default=10)
    parser.add_argument('--method', choices=['highs', 'highs-ds', 'highs-ipm'], default='highs-ds')
    parser.add_argument('--algorithm', choices=['efficient', 'full', 'both'], default='efficient')
    parser.add_argument('--presolve', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--apply-sigmoid-transform', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--prune-redundant', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--cutting-plane-tol', type=float, default=1e-8)
    parser.add_argument('--cutting-plane-max-iter', type=int, default=50)
    parser.add_argument('--cutting-plane-max-add', type=int, default=5000)
    parser.add_argument('--cutting-plane-warm-start', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--a-list', nargs='+', type=float, default=None)
    parser.add_argument('--a-min', type=float, default=-5.0)
    parser.add_argument('--a-max', type=float, default=5.0)
    parser.add_argument('--a-step', type=float, default=1.0)
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    if args.mode == 'list-cases':
        print('Available cases:')
        for case_name in list_case_names():
            case = CASE_REGISTRY[case_name]
            print(f'- {case.name}: {case.description} | truth={case.truth}')
        print('- skew_power_sweep: generated by --mode run-skew-power')
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    common_kwargs = dict(
        n_list=args.n_list,
        B_list=args.B_list,
        n_rep=args.n_rep,
        alpha=args.alpha,
        seed=args.seed,
        outdir=outdir,
        progress_every=args.progress_every,
        method=args.method,
        presolve=args.presolve,
        apply_sigmoid_transform=args.apply_sigmoid_transform,
        prune_redundant=args.prune_redundant,
        algorithm=args.algorithm,
        cutting_plane_tol=args.cutting_plane_tol,
        cutting_plane_max_iter=args.cutting_plane_max_iter,
        cutting_plane_max_add=args.cutting_plane_max_add,
        cutting_plane_warm_start=args.cutting_plane_warm_start,
    )

    if args.mode == 'run-skew-power':
        a_values = _make_a_grid(args.a_list, args.a_min, args.a_max, args.a_step)
        combined_detail, combined_summary = run_skew_power_sweep(a_values=a_values, **common_kwargs)
        cases_for_meta = [f'skew_power_a={a:g}' for a in a_values]
    else:
        cases = args.cases if args.mode == 'run-case' else DEFAULT_CASES
        combined_detail, combined_summary = run_suite(cases=cases, **common_kwargs)
        cases_for_meta = list(cases)

    meta = {
        'mode': args.mode,
        'cases': cases_for_meta,
        'n_list': args.n_list,
        'B_list': args.B_list,
        'n_rep': args.n_rep,
        'alpha': args.alpha,
        'seed': args.seed,
        'algorithm': args.algorithm,
        'method': args.method,
        'presolve': args.presolve,
        'apply_sigmoid_transform': args.apply_sigmoid_transform,
        'prune_redundant': args.prune_redundant,
        'cutting_plane_tol': args.cutting_plane_tol,
        'cutting_plane_max_iter': args.cutting_plane_max_iter,
        'cutting_plane_max_add': args.cutting_plane_max_add,
        'cutting_plane_warm_start': args.cutting_plane_warm_start,
        'combined_detail_rows': int(len(combined_detail)),
        'combined_summary_rows': int(len(combined_summary)),
    }
    with open(outdir / 'run_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('\nDone. Combined summary:')
    keep_cols = [
        'case',
        'algorithm',
        'n',
        'B',
        'reject_rate',
        'mean_p_value',
        'mean_statistic',
        'mean_runtime_sec',
    ]
    if 'a' in combined_summary.columns:
        keep_cols.insert(1, 'a')
    print(combined_summary[keep_cols])
    print(f'\nResults saved to: {outdir.resolve()}')


if __name__ == '__main__':
    main()
