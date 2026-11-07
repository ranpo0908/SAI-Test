"""Create summaries, LaTeX tables, and PDF plots from completed experiment files."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


STYLES = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D")]
FIGURES = {"gaussian_near": "plot_alt_var_reject_vs_n",
           "gaussian_path": "path_rejection", "skew_path": "skew_power_rejection_plot"}


def summarize(detail):
    return detail.groupby(["study", "case", "n", "B", "a", "alpha", "solver"],
                          dropna=False, as_index=False).agg(
        n_rep=("rep", "size"), reject_rate=("reject", "mean"),
        mean_p_value=("p_value", "mean"), mean_statistic=("statistic", "mean"),
        mean_critical_value=("critical_value", "mean"),
        mean_runtime_sec=("runtime_sec", "mean"), max_violation=("max_violation", "max"))


def _format(value):
    if isinstance(value, (float, np.floating)):
        return f"{value:.6f}"
    return str(value)


def _markdown_table(frame):
    header = "| " + " | ".join(map(str, frame.columns)) + " |\n"
    header += "| " + " | ".join(["---"] * len(frame.columns)) + " |\n"
    return header + "".join("| " + " | ".join(_format(v) for v in row) + " |\n"
                            for row in frame.itertuples(index=False, name=None))


def _latex_table(frame, caption, label, path):
    header = " & ".join(frame.columns) + r" \\"
    rows = [" & ".join(_format(value) for value in row) + r" \\"
            for row in frame.itertuples(index=False, name=None)]
    text = ("\\begin{table}[htbp]\n\\centering\n"
            + "\\caption{" + caption + "}\n\\label{" + label + "}\n"
            + "\\begin{tabular}{" + "c" * len(frame.columns) + "}\n\\hline\n"
            + header + "\n\\hline\n" + "\n".join(rows)
            + "\n\\hline\n\\end{tabular}\n\\end{table}\n")
    path.write_text(text, encoding="utf-8")


def manuscript_tables(summary, outdir, quick):
    outputs = []
    selected = summary.loc[summary.study == "gaussian_table"]
    if not selected.empty:
        rows = []
        for (n, B), group in selected.groupby(["n", "B"], sort=True):
            by_case = group.set_index("case")
            null, alt = by_case.loc["gaussian_null"], by_case.loc["gaussian_alt"]
            rows.append([n, B, null.reject_rate, null.mean_p_value,
                         alt.reject_rate, alt.mean_p_value])
        frame = pd.DataFrame(rows, columns=[r"$n$", r"$B$", "Null rejection", "Null mean $p$",
                                           "Alt. rejection", "Alt. mean $p$"])
        outputs.append(("table1_gaussian", frame, "Performance of the SAI test for Gaussian distributions.",
                        "tab:fixed-two-cases"))
    selected = summary.loc[summary.study == "skew_table"]
    if not selected.empty:
        rows = []
        for case, label in [("skew_null", "(i) Null"), ("skew_alt", "(ii) Alternative")]:
            for row in selected.loc[selected.case == case].sort_values(["n", "B"]).itertuples():
                rows.append([label, row.n, row.B, row.reject_rate, row.mean_p_value, row.mean_statistic])
        frame = pd.DataFrame(rows, columns=["Case", r"$n$", r"$B$", "Rejection rate", "Mean $p$", "Mean statistic"])
        outputs.append(("table2_skew", frame, "Performance of the SAI test for skew-normal distributions.",
                        "tab:skew-obvious"))
    selected = summary.loc[summary.study == "skew_near"]
    if not selected.empty:
        rows = [[r.n, r.B, r.reject_rate, r.mean_p_value, r.mean_statistic]
                for r in selected.sort_values(["B", "n"]).itertuples()]
        frame = pd.DataFrame(rows, columns=[r"$n$", r"$B$", "Rejection rate", "Mean $p$", "Mean statistic"])
        outputs.append(("table3_skew_near", frame, "Performance for the weaker skew-normal alternative.",
                        "tab:hard-to-tell"))
    for name, frame, caption, label in outputs:
        if quick:
            name += "_quick"
            caption = "Quick validation run. " + caption
        frame.to_csv(outdir / f"{name}.csv", index=False)
        _latex_table(frame, caption, label, outdir / f"{name}.tex")
    return [name + ("_quick" if quick else "") for name, *_ in outputs]


def make_plots(summary, outdir, quick):
    outputs = []
    with plt.rc_context({"font.family": "serif", "font.size": 10,
                         "mathtext.fontset": "dejavuserif", "axes.labelsize": 11,
                         "legend.fontsize": 9, "pdf.fonttype": 42, "axes.linewidth": 0.8}):
        for study, filename in FIGURES.items():
            selected = summary.loc[summary.study == study]
            if selected.empty:
                continue
            if selected.B.nunique() != 1:
                raise ValueError(f"{study}: a plot must use one B setting.")
            fig, ax = plt.subplots(figsize=(4.8, 3.3), layout="constrained")
            if study == "gaussian_near":
                points = selected.sort_values("n")
                ax.plot(points.n, points.reject_rate, color="black", linestyle="-",
                        marker="o", linewidth=1.4, markersize=4.5)
                ax.set_xlabel(r"Sample size $n$")
                ax.set_xticks(points.n)
            else:
                sizes = sorted(selected.n.unique())
                if len(sizes) > len(STYLES):
                    raise ValueError("Choose at most four sample sizes for a readable figure.")
                for n, (linestyle, marker) in zip(sizes, STYLES):
                    points = selected.loc[selected.n == n].sort_values("a")
                    ax.plot(points.a, points.reject_rate, color="black", linestyle=linestyle,
                            marker=marker, markerfacecolor="white", markeredgewidth=1,
                            linewidth=1.4, markersize=4.5, label=rf"$n={int(n)}$")
                ax.set_xlabel(r"Location parameter $a$")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
                ax.legend(loc="best", frameon=False)
            ax.set_ylabel("Empirical rejection rate")
            ax.set_ylim(-0.025, 1.025)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(direction="out", length=3)
            ax.margins(x=0.04)
            if quick:
                filename += "_quick"
                ax.set_title("Quick validation run", fontsize=10)
            path = outdir / f"{filename}.pdf"
            fig.savefig(path, facecolor="white")
            plt.close(fig)
            outputs.append(path.name)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    metadata = json.loads((args.results / "run_config.json").read_text(encoding="utf-8"))
    settings = metadata["settings"]
    detail = pd.read_csv(args.results / "detail.csv", dtype={"data_seed": "uint64", "mc_seed": "uint64"})
    keys = ["study", "case", "n", "B", "a", "rep"]
    if detail.duplicated(keys).any():
        raise ValueError("Duplicate repetitions found; no report was written.")
    if len(detail) != metadata["expected_repetitions"]:
        raise ValueError("Run is incomplete. Resume the experiments before making the report.")
    for name, study in settings["studies"].items():
        subset = detail.loc[detail.study == name]
        count = len(study["cases"]) * len(study["n"]) * len(study["B"]) * len(study["a"])
        if len(subset) != count * settings["repetitions"]:
            raise ValueError(f"{name}: unexpected experiment count.")
        for column in ("case", "n", "B"):
            allowed = study["cases"] if column == "case" else study[column]
            if not subset[column].isin(allowed).all():
                raise ValueError(f"{name}: unexpected values of {column}.")
        if study["a"] == [None]:
            if subset.a.notna().any():
                raise ValueError(f"{name}: unexpected path parameters.")
        elif not subset.a.isin(study["a"]).all():
            raise ValueError(f"{name}: unexpected path parameters.")
    summary = summarize(detail)
    if not summary.n_rep.eq(settings["repetitions"]).all():
        raise ValueError("Some configurations have an incorrect repetition count.")
    if not detail.p_value.between(0, 1).all() or not detail.reject.isin([0, 1]).all():
        raise ValueError("Invalid p-values or rejection indicators.")
    if (not detail.rep.isin(range(1, settings["repetitions"] + 1)).all()
            or not detail.alpha.eq(settings["alpha"]).all()
            or not detail.solver.eq(settings["solver"]).all()
            or not detail["transform"].eq(1).all()):
        raise ValueError("Rows do not match the saved run settings.")
    if not detail.reject.eq((detail.p_value <= settings["alpha"]).astype(int)).all():
        raise ValueError("Rejection indicators disagree with the p-values.")
    if not detail.max_violation.between(0, settings["constraint_tolerance"]).all():
        raise ValueError("Some LP solutions exceed the saved feasibility tolerance.")
    outdir = args.results / "report"
    tables_dir, figures_dir = outdir / "tables", outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    quick = settings["run_type"] == "quick_validation"
    summary.to_csv(outdir / "summary.csv", index=False)
    tables = manuscript_tables(summary, tables_dir, quick)
    figures = make_plots(summary, figures_dir, quick)
    heading = "Quick validation report - not manuscript results" if quick else "SAI experiment report"
    report = [f"# {heading}\n", f"Repetitions per configuration: {settings['repetitions']}\n",
              f"Significance level: {settings['alpha']}; solver: {settings['solver']}\n",
              f"Total repetitions: {len(detail)}; configurations: {len(summary)}\n",
              "Raw results and both RNG seeds are retained in `detail.csv`.\n",
              "## Configuration summaries\n", _markdown_table(summary), "\n## Tables\n"]
    report += [f"- [{name}.tex](tables/{name}.tex)\n" for name in tables]
    report += ["\n## Figures\n"] + [f"- [{name}](figures/{name})\n" for name in figures]
    (outdir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Report: {(outdir / 'report.md').resolve()}")
    for name in figures:
        print(f"PDF: {(figures_dir / name).resolve()}")


if __name__ == "__main__":
    main()
