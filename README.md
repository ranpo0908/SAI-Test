# SAI-Test

Monte Carlo testing of the stochastic arrangement increasing (SAI) property for
bivariate distributions using a Lipschitz discrepancy. The project includes
normal and skew-normal simulation studies, repetition-level results, tables,
and PDF figures.

The test statistic is computed by a sparse linear program. Both the full LP and
the optional cutting-plane implementation use SciPy's HiGHS dual-simplex solver.

## Project structure

| File | Purpose |
| --- | --- |
| `scripts/construct_test.py` | Construct and solve the LP; compute the statistic, Monte Carlo p-value, and rejection decision. |
| `scripts/generate_samples.py` | Generate bivariate normal and skew-normal samples; define the experiment distributions. |
| `scripts/run_experiments.py` | Run selected studies and save one row for every repetition. |
| `scripts/make_report.py` | Read completed results and generate summaries, tables, and PDF figures. |
| `requirement.txt` | Python dependencies. |

Run all commands below from the repository root, which contains `scripts/`.

## Installation

Use Python 3.10 or newer. Install the dependencies with:

```bash
python -m pip install -r requirement.txt
```

The four runtime dependencies are NumPy, SciPy, pandas, and Matplotlib.

## Quick check

```bash
python scripts/run_experiments.py --quick --workers 2 --outdir results_quick
python scripts/make_report.py --results results_quick
```

This runs all six studies on smaller grids: sample sizes 20 and 40, 19 Monte
Carlo draws, two repetitions, and shortened location paths. It produces 48
repetitions in total. Reports and figure filenames are marked as quick results.

Run the report command after the experiment command finishes successfully.

## Experiment designs

Here, `n` is the sample size, `B` is the number of Monte Carlo draws per test,
and `a` is the location parameter varied along a distribution path.

| Study | Description | Sample sizes `n` | Monte Carlo draws `B` | Location path `a` |
| --- | --- | --- | --- | --- |
| `gaussian_table` | Normal null and clear alternative | 50, 100, 200 | 19, 99, 199, 499 | Fixed |
| `gaussian_near` | Normal unequal-variance alternative | 50, 150, 300, 500 | 199 | Fixed |
| `gaussian_path` | Normal transition comparison | 50, 100, 200 | 199 | -1 to 1, step 0.2 |
| `skew_table` | Skew-normal null and clear alternative | 50, 100, 200 | 99, 199 | Fixed |
| `skew_near` | Weaker skew-normal alternative | 50, 100, 200, 300 | 99, 199 | Fixed |
| `skew_path` | Skew-normal transition comparison | 100, 200 | 99 | -5 to 5, step 1 |

Distribution parameters are defined in `scripts/generate_samples.py`. The
grids are defined in `STUDIES` in `scripts/run_experiments.py`. For skew-normal
distributions, `scale` denotes the scale matrix; it generally differs from the
covariance matrix.

### Run all studies

```bash
python scripts/run_experiments.py --study all --n-rep 50 --workers 4 --outdir results_all
python scripts/make_report.py --results results_all
```

The complete design has 103 configurations and 5,150 repetitions. Defaults are
50 repetitions per configuration, significance level 0.05, base seed 20260416,
and the full sparse LP solver. The experiment runner applies the same logistic
transform to both coordinates.

### Run normal studies

```bash
python scripts/run_experiments.py \
  --study gaussian_table gaussian_near gaussian_path \
  --n-rep 50 --workers 4 --outdir results_gaussian
python scripts/make_report.py --results results_gaussian
```

Including `gaussian_path` produces the transition figure comparing sample sizes
50, 100, and 200. The three normal studies contain 3,050 repetitions at the
default repetition count.

### Run skew-normal studies

```bash
python scripts/run_experiments.py \
  --study skew_table skew_near skew_path \
  --n-rep 50 --workers 4 --outdir results_skew_new
python scripts/make_report.py --results results_skew_new
```

The three skew-normal studies contain 2,100 repetitions at the default count.
Choose a fresh output directory for each new run.

## Included results

The saved experiments use 50 repetitions per configuration and are arranged as
follows:

| Directory | Studies | Repetitions |
| --- | --- | --- |
| `results_normal/` | `gaussian_table`, `gaussian_near` | 1,400 |
| `results_normal_path/` | `gaussian_path` | 1,650 |
| `results_skew/` | `skew_table`, `skew_near`, `skew_path` | 2,100 |

Rebuild their reports from the saved data with:

```bash
python scripts/make_report.py --results results_normal
python scripts/make_report.py --results results_normal_path
python scripts/make_report.py --results results_skew
```

Report generation reads the saved results without running new simulations.

## Output files

Each experiment directory contains:

| Path | Contents |
| --- | --- |
| `detail.csv` | Every repetition, including both random seeds, statistic, critical value, p-value, rejection decision, runtime, and maximum LP constraint violation. |
| `run_config.json` | Study grids, distribution parameters, seed, solver settings, tolerances, NumPy/SciPy versions, and source hashes. |
| `report/summary.csv` | Rejection rates and other summaries for each configuration. |
| `report/tables/` | Tables in CSV and LaTeX formats. |
| `report/figures/` | Vector PDF figures. |
| `report/report.md` | A report linking the tables and figures. |

The figure names are:

| Study | PDF filename |
| --- | --- |
| `gaussian_near` | `plot_alt_var_reject_vs_n.pdf` |
| `gaussian_path` | `path_rejection.pdf` |
| `skew_path` | `skew_power_rejection_plot.pdf` |

Figures use black lines. Distinct line styles and markers distinguish curves.
Quick runs add `_quick` to table and figure basenames.

## Reproducibility and resuming

Data generation and Monte Carlo calibration use separate seeds, both saved in
`detail.csv`. The base seed can be changed with `--seed`. For a given study
configuration and repetition, changing the worker count or solver selection
preserves the generated sample and Monte Carlo signs when the other settings
and numerical-library versions are fixed.

Saved rows are flushed after each repetition is written. Resume an interrupted
run with the same arguments and `--resume`, for example:

```bash
python scripts/run_experiments.py --study all --n-rep 50 --workers 4 --outdir results_all --resume
```

Resuming requires matching experiment settings, source hashes, and NumPy/SciPy
versions. The worker count may change. Reports require a complete set of
repetitions for the saved experiment plan.

To use the cutting-plane implementation, pass `--solver cutting-plane` when
starting a run in a fresh directory. Both solver choices compute the same LP
formulation and check feasibility against all original constraints.

## Apply the test directly

From the repository root:

```python
import numpy as np

from scripts.construct_test import monte_carlo_lipsai_test
from scripts.generate_samples import sample_bivariate_normal

x, y = sample_bivariate_normal(
    n=100,
    rng=np.random.default_rng(123),
    location=[0, 1],
    scale=[[1, 0.3], [0.3, 1]],
)
result = monte_carlo_lipsai_test(
    x, y, B=199, alpha=0.05,
    rng=np.random.default_rng(456),
    solver="full",
)
print(result["statistic"], result["p_value"], result["reject"])
```

For your own data, supply the paired observations as two finite, nonempty,
one-dimensional arrays of equal length. The function applies the logistic
transform by default. Use `transform=False` when the data are already on the
bounded scale you intend to test.

The result also contains `critical_value`, `mc_values`, and `max_violation`.
Rejection uses the plus-one Monte Carlo p-value, including ties, at the chosen
significance level.

For all command-line options:

```bash
python scripts/run_experiments.py --help
python scripts/make_report.py --help
```
