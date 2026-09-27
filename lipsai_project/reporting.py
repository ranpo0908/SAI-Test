from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd



def write_markdown_table(df: pd.DataFrame, path: Path, float_digits: int = 4) -> None:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: f'{x:.{float_digits}f}')
    header = '|' + '|'.join(out.columns.astype(str)) + '|\n'
    sep = '|' + '|'.join(['---'] * len(out.columns)) + '|\n'
    rows = ['|' + '|'.join(map(str, row)) + '|\n' for row in out.to_numpy()]
    path.write_text(header + sep + ''.join(rows), encoding='utf-8')



def _optional_float_line(summary: Dict[str, object], key: str, label: str, digits: int = 4) -> str:
    if key not in summary:
        return ''
    return f"\n{label:<20}: {float(summary[key]):.{digits}f}"



def format_summary_block(summary: Dict[str, object]) -> str:
    base = (
        f"case               : {summary['case']}\n"
        f"description        : {summary['description']}\n"
        f"truth label        : {summary['truth']}\n"
    )
    if 'a' in summary:
        base += f"a                  : {float(summary['a']):g}\n"
    if 'algorithm' in summary:
        base += f"algorithm          : {summary['algorithm']}\n"
    base += (
        f"n                  : {int(float(summary['n']))}\n"
        f"B                  : {int(float(summary['B']))}\n"
        f"n_rep              : {int(float(summary['n_rep']))}\n"
        f"reject_rate        : {float(summary['reject_rate']):.4f}\n"
        f"mean_p_value       : {float(summary['mean_p_value']):.4f}\n"
        f"median_p_value     : {float(summary['median_p_value']):.4f}\n"
        f"mean_statistic     : {float(summary['mean_statistic']):.4f}\n"
        f"mean_critical_value: {float(summary['mean_critical_value']):.4f}\n"
        f"mean_runtime_sec   : {float(summary['mean_runtime_sec']):.4f}"
    )
    base += _optional_float_line(summary, 'mean_n_full_constraints', 'mean_full_constraints', 1)
    base += _optional_float_line(summary, 'mean_obs_constraints', 'mean_obs_constraints', 1)
    base += _optional_float_line(summary, 'mean_mc_constraints', 'mean_mc_constraints', 1)
    base += _optional_float_line(summary, 'mean_obs_iterations', 'mean_obs_iterations', 2)
    base += _optional_float_line(summary, 'mean_mc_iterations', 'mean_mc_iterations', 2)
    return base



def save_experiment_outputs(
    *,
    outdir: Path,
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    basename: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    detail_path = outdir / f'{basename}_detail.csv'
    summary_path = outdir / f'{basename}_summary.csv'
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_markdown_table(summary_df, outdir / f'{basename}_summary.md')

    meta = {
        'basename': basename,
        'detail_file': detail_path.name,
        'summary_file': summary_path.name,
        'rows_detail': int(len(detail_df)),
        'rows_summary': int(len(summary_df)),
    }
    with open(outdir / f'{basename}_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
