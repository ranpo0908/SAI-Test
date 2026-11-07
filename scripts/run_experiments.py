"""Run manuscript experiments and save one row per repetition. No plotting."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

# Avoid nested BLAS thread pools when using independent worker processes.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy

from construct_test import monte_carlo_lipsai_test, FEASIBILITY_TOL, CONSTRAINT_TOL
from generate_samples import sample_case, case_parameters, CASE_PARAMETERS


# The six entries correspond to Tables 1-3 and Figures 1-3 in the manuscript.
STUDIES = {
    "gaussian_table": {"cases": ["gaussian_null", "gaussian_alt"],
                       "n": [50, 100, 200], "B": [19, 99, 199, 499], "a": [None]},
    "gaussian_near": {"cases": ["gaussian_near"],
                      "n": [50, 150, 300, 500], "B": [199], "a": [None]},
    "gaussian_path": {"cases": ["gaussian_path"], "n": [50, 100, 200],
                      "B": [199], "a": [round(k / 5, 8) for k in range(-5, 6)]},
    "skew_table": {"cases": ["skew_null", "skew_alt"],
                   "n": [50, 100, 200], "B": [99, 199], "a": [None]},
    "skew_near": {"cases": ["skew_near"],
                  "n": [50, 100, 200, 300], "B": [99, 199], "a": [None]},
    "skew_path": {"cases": ["skew_path"], "n": [100, 200],
                  "B": [99], "a": list(range(-5, 6))},
}

FIELDS = ["study", "case", "n", "B", "a", "rep", "data_seed", "mc_seed",
          "alpha", "solver", "transform", "statistic", "critical_value",
          "p_value", "reject", "runtime_sec", "max_violation"]


def experiment_plan(selected, repetitions, quick=False):
    """A plain configuration dictionary controls the single experiment loop."""
    plan = json.loads(json.dumps({name: STUDIES[name] for name in selected}))
    if quick:
        for name, study in plan.items():
            study["n"], study["B"] = [20, 40], [19]
            if name == "gaussian_path":
                study["a"] = [-1, 0, 1]
            elif name == "skew_path":
                study["a"] = [-2, -1, 0]
    return {"studies": plan, "repetitions": repetitions}


def make_jobs(plan, seed, solver, alpha):
    jobs = []
    for study_name, study in plan["studies"].items():
        study_id = list(STUDIES).index(study_name)
        for case_id, case in enumerate(study["cases"]):
            for n in study["n"]:
                for B in study["B"]:
                    for a_id, a in enumerate(study["a"]):
                        for rep in range(1, plan["repetitions"] + 1):
                            # Solver is deliberately excluded: both solvers get matched draws.
                            seeds = np.random.SeedSequence(
                                [seed, study_id, case_id, n, B, a_id, rep]
                            ).generate_state(2, dtype=np.uint64)
                            jobs.append({"study": study_name, "case": case, "n": n,
                                         "B": B, "a": a, "rep": rep,
                                         "data_seed": int(seeds[0]), "mc_seed": int(seeds[1]),
                                         "alpha": alpha, "solver": solver, "transform": 1})
    return jobs


def _row_key(row):
    a = row["a"]
    a = None if a in (None, "") else float(a)
    return (row["study"], row["case"], int(row["n"]), int(row["B"]), a, int(row["rep"]))


def run_one(job):
    x, y = sample_case(job["case"], job["n"],
                       np.random.default_rng(job["data_seed"]), a=job["a"])
    start = time.perf_counter()
    result = monte_carlo_lipsai_test(x, y, B=job["B"], alpha=job["alpha"],
                                    rng=np.random.default_rng(job["mc_seed"]),
                                    solver=job["solver"], transform=True)
    return {**job, "statistic": result["statistic"],
            "critical_value": result["critical_value"], "p_value": result["p_value"],
            "reject": int(result["reject"]), "runtime_sec": time.perf_counter() - start,
            "max_violation": result["max_violation"]}


def _write_rows(rows, path, completed, total):
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        for count, row in enumerate(rows, completed + 1):
            writer.writerow(row)
            stream.flush()  # Each completed repetition is retained immediately.
            if count % 25 == 0 or count == total:
                print(f"Completed {count}/{total}: {row['study']}, n={row['n']}, B={row['B']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", nargs="+", choices=["all", *STUDIES], default=["all"])
    parser.add_argument("--n-rep", type=int, default=None,
                        help="Default: 50 for manuscript runs; 2 with --quick.")
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--solver", choices=["full", "cutting-plane"], default="full")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true", help="Small validation grids, not manuscript results.")
    parser.add_argument("--resume", action="store_true", help="Resume an identical saved configuration.")
    args = parser.parse_args()
    repetitions = args.n_rep if args.n_rep is not None else (2 if args.quick else 50)
    if repetitions < 1 or args.workers < 1 or args.seed < 0 or not 0 < args.alpha < 1:
        parser.error("Repetitions and workers must be positive; seed nonnegative; alpha in (0,1).")
    if "all" in args.study and len(args.study) != 1:
        parser.error("Use either --study all or a list of individual studies.")
    selected = list(STUDIES) if args.study == ["all"] else list(dict.fromkeys(args.study))
    plan = experiment_plan(selected, repetitions, args.quick)
    settings = {**plan, "base_seed": args.seed, "alpha": args.alpha,
                "solver": args.solver, "transform": "logistic",
                "run_type": "quick_validation" if args.quick else "manuscript",
                "distribution_parameters": CASE_PARAMETERS,
                "path_parameters": {
                    name: [{"a": a, **case_parameters(name, a)} for a in study["a"]]
                    for name, study in plan["studies"].items() if name.endswith("_path")},
                "feasibility_tolerance": FEASIBILITY_TOL, "constraint_tolerance": CONSTRAINT_TOL}
    jobs = make_jobs(plan, args.seed, args.solver, args.alpha)
    source_dir = Path(__file__).resolve().parent
    versions = {"numpy": np.__version__, "scipy": scipy.__version__}
    source_hashes = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                     for name in ("construct_test.py", "generate_samples.py", "run_experiments.py")}
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path, config_path = args.outdir / "detail.csv", args.outdir / "run_config.json"
    completed_keys = set()
    if csv_path.exists():
        if not args.resume:
            parser.error("Results already exist. Use --resume or choose a new --outdir.")
        saved = json.loads(config_path.read_text()) if config_path.exists() else {}
        if saved.get("settings") != settings:
            parser.error("Saved settings differ from this request. Choose a new --outdir.")
        if saved.get("source_sha256") != source_hashes or saved.get("versions") != versions:
            parser.error("Code or numerical-library versions changed. Choose a new --outdir.")
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        completed_keys = {_row_key(row) for row in rows}
        if len(completed_keys) != len(rows):
            parser.error("Existing result file contains duplicate repetitions.")
        expected_keys = {_row_key(job) for job in jobs}
        if not completed_keys <= expected_keys:
            parser.error("Existing rows do not belong to the saved experiment plan.")
    metadata = {"settings": settings, "expected_repetitions": len(jobs),
                "versions": versions, "source_sha256": source_hashes}
    config_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pending = [job for job in jobs if _row_key(job) not in completed_keys]
    print(f"Run type: {settings['run_type']}; {len(pending)} pending / {len(jobs)} total repetitions", flush=True)
    if args.workers == 1:
        _write_rows(map(run_one, pending), csv_path, len(completed_keys), len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            _write_rows(pool.map(run_one, pending), csv_path, len(completed_keys), len(jobs))
    print(f"Results: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
