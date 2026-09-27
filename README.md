# Lip-SAI experiment project, v2

This version keeps the original finite-dimensional Lip-SAI Monte Carlo workflow, but adds a separate efficient solver path for performance comparison.

## Main files

- `lipsai_project/pipeline.py`: LP construction, full sparse LP solver, efficient cutting-plane solver, Monte Carlo p-value pipeline, and timing/constraint metrics.
- `lipsai_project/distributions.py`: registry of bivariate normal and bivariate skew-normal cases, plus a generated skew-normal power-sweep case.
- `lipsai_project/reporting.py`: summary formatting and output saving.
- `run_lipsai_project.py`: command-line entry point based on `argparse`.

## Algorithms

The default is the efficient algorithm:

```bash
python run_lipsai_project.py --mode run-suite --algorithm efficient
```

Use the original/full LP path for comparison:

```bash
python run_lipsai_project.py --mode run-suite --algorithm full
```

Run both algorithms on the same grid:

```bash
python run_lipsai_project.py --mode run-case --cases skew_14 --n-list 100 200 --B-list 99 --n-rep 20 --algorithm both
```

### Efficient method

`--algorithm efficient` uses delayed constraint generation / cutting plane:

1. Build the pairwise geometry once.
2. Solve a restricted LP with only active constraints.
3. Vectorize the search over all omitted Lipschitz and antisymmetry constraints.
4. Add the most violated constraints.
5. Repeat until the maximum violation is below `--cutting-plane-tol`.

Useful options:

```bash
--cutting-plane-tol 1e-8
--cutting-plane-max-iter 50
--cutting-plane-max-add 5000
--cutting-plane-warm-start / --no-cutting-plane-warm-start
```

## Available fixed cases

```bash
python run_lipsai_project.py --mode list-cases
```

Fixed cases:

- `null_l_sai`
- `alt_mean_cov`
- `alt_var`
- `skew_14`
- `skew_41`
- `skew_11`

## Basic commands

Run the default full suite with the efficient method:

```bash
python run_lipsai_project.py --mode run-suite
```

Run just the two skew-normal examples:

```bash
python run_lipsai_project.py --mode run-case --cases skew_14 skew_41 --n-list 50 100 200 --B-list 99 199 --n-rep 30
```

Turn off the sigmoid transform:

```bash
python run_lipsai_project.py --mode run-case --cases skew_14 --no-apply-sigmoid-transform
```

## New skew-normal power sweep

This runs

```text
SN([0,a], [[1,0.3],[0.3,4]], [1,4])
```

with `a` moving from `-5` to `5` by default.

```bash
python run_lipsai_project.py --mode run-skew-power --n-list 100 200 --B-list 99 --n-rep 30
```

Customize the grid:

```bash
python run_lipsai_project.py --mode run-skew-power --a-min -5 --a-max 5 --a-step 0.5 --n-list 200 --B-list 199 --n-rep 50
```

Or give explicit values:

```bash
python run_lipsai_project.py --mode run-skew-power --a-list -5 -3 -1 0 1 3 5 --n-list 200 --B-list 199 --n-rep 50
```

## Output metrics

The detail and summary CSV files now include performance diagnostics, including:

- `runtime_sec`
- `geometry_build_sec`
- `matrix_build_sec`
- `lp_solve_sec`
- `separation_sec`
- `n_full_constraints`
- `obs_constraints`
- `mean_mc_constraints`
- `obs_iterations`
- `mean_mc_iterations`
- `max_violation`

These metrics are meant for later comparison between `efficient`, `full`, and `both` runs.

## Note on the skew-normal generator

SciPy provides a built-in univariate skew-normal distribution, but not a built-in multivariate skew-normal sampler in `scipy.stats`. The bivariate skew-normal sampler here is implemented directly in `distributions.py` using the standard Azzalini-type stochastic representation.
