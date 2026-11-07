"""Construct the finite SAI LP and its Monte Carlo test. No sampling or file I/O."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import expit


FEASIBILITY_TOL = 1e-9
CONSTRAINT_TOL = 1e-8
CUT_BATCH_SIZE = 5000


# "dataclass" is a class decorator from Python's standard library.
# A decorator receives the class after Python has created it and can add methods.
# Thus, @dataclass here is approximately: LPProblem = dataclass(LPProblem).
#
# With its default settings, @dataclass generates methods including:
#   * __init__: initialize the fields declared below;
#   * __repr__: show their names and values when the object is printed;
#   * __eq__: compare corresponding field values.
# We do not need to write these routine methods ourselves.
#
# IMPORTANT: the generated __eq__ is not useful for comparing this record's
# multi-element NumPy arrays. Array equality is elementwise and can produce an
# ambiguous-truth-value error. Compare particular arrays with np.array_equal or
# np.allclose when checking two LP problems. The solver never uses lp1 == lp2.
#
# This is a mutable dataclass (the default frozen=False). That is intentional:
# build_lp later fills the full-matrix cache, and solve_lp records its reusable
# constraints and feasibility diagnostic. Treat the original geometry/bounds as
# fixed once the problem has been constructed.
@dataclass
class LPProblem:
    """Store one sample's LP data and the caches reused across objective vectors.

    Mathematical role
    -----------------
    The unknown vector is v = (v_0, ..., v_{n-1}), where
    v_k = u(y_k, x_k) in the manuscript's antisymmetric formulation. We solve

        maximize    weights @ v
        subject to  lower <= v <= upper
                    all pairwise Lipschitz/antisymmetry inequalities.

    The sample fixes the feasible set. Only weights changes between the
    observed statistic and the Monte Carlo draws. Consequently, weights are
    passed to solve_lp rather than stored in this record.

    All coordinate formulas below refer to the x and y supplied to build_lp.
    The Monte Carlo test applies the logistic transform first by default.

    What @dataclass does here
    -------------------------
    Each annotated attribute below is a dataclass field. A declaration such as
    "lower: np.ndarray" describes a required constructor argument. A declaration
    such as "A: csr_matrix | None = None" also supplies a default value.
    The generated constructor has the following argument order:

        LPProblem(lower, upper, i, j, d_same, d_flip, solver,
                  A=None, b=None, initial_constraints=None,
                  max_violation=0.0)

    It behaves essentially like assignments of the form "self.lower = lower".
    The seven fields without defaults are required; the last four are optional.
    Fields without defaults must appear before fields with defaults, just as
    required function arguments precede optional arguments.

    Type annotations document expected types. They do not allocate arrays,
    convert inputs, check array shapes, or enforce types at runtime. The input
    validation and numerical construction happen in build_lp and its helpers.
    The dataclass does not build a matrix or run an optimization by itself.

    Arrays are stored by reference, not copied automatically. If an external
    variable refers to the same array, an in-place edit through either reference
    changes the array held here. The construction functions create the intended
    arrays; normally callers should obtain this object through build_lp.

    Dimensions and solver modes
    ---------------------------
    Let n be the sample size and m = n*(n-1)//2 the number of unordered pairs.
    lower and upper have length n. The four geometry arrays i, j, d_same, and
    d_flip have length m. For n=1 these geometry arrays are empty, which is valid.

    In full mode, build_lp fills A and b once. Every objective reuses them.
    In cutting-plane mode, A and b stay None; each restricted matrix is local
    to the solver. The geometry and an initial working set are reused instead.
    """

    # Componentwise lower limits on the n unknown function values.
    # Define radius_k = abs(x_k-y_k)/sqrt(2). Then:
    #   lower[k] = -radius_k if x_k < y_k;
    #   lower[k] = 0         otherwise.
    # Shape: (n,). Expected entries: finite floating-point numbers.
    lower: np.ndarray

    # Componentwise upper limits, with the same ordering as lower.
    #   upper[k] = radius_k if x_k > y_k;
    #   upper[k] = 0        otherwise.
    # Thus x_k>y_k gives 0<=v_k<=radius_k, while x_k<y_k gives
    # -radius_k<=v_k<=0. If x_k=y_k, both bounds are zero.
    # These bounds encode the sign restriction and the diagonal Lipschitz bound.
    # The zero vector is always allowed. The finite bounds keep every restricted
    # LP bounded even before pairwise constraints have been added.
    # Shape: (n,). Expected entries: finite floating-point numbers.
    upper: np.ndarray

    # First observation index for each unordered pair. Python indices start at 0.
    # Constructed together with j by np.triu_indices(n, k=1), so i[p] < j[p].
    # For n=3: i = [0, 0, 1], j = [1, 2, 2].
    # Here p indexes a pair; i[p] and j[p] index observations/LP variables.
    # Using i<j stores each unordered pair once, without diagonal duplicates.
    # Shape: (m,). Expected entries: integers.
    i: np.ndarray

    # Second observation index for the same pairs and in the same order as i.
    # The p-th pair is (i[p], j[p]); changing their ordering independently would
    # make the cached distances correspond to the wrong constraints.
    # Shape: (m,). Expected entries: integers.
    j: np.ndarray

    # Distance between the two observations in their original orientation:
    #   d_same[p] = sqrt((x_i-x_j)**2 + (y_i-y_j)**2),
    # where i=i[p] and j=j[p]. It is the right-hand side of
    #   |v_i-v_j| <= d_same[p].
    # That absolute-value inequality supplies two linear inequalities.
    # Shape: (m,). Expected entries: nonnegative floating-point numbers.
    d_same: np.ndarray

    # Distance after exchanging the coordinates of the second observation:
    #   d_flip[p] = sqrt((x_i-y_j)**2 + (y_i-x_j)**2).
    # It is the right-hand side of |v_i+v_j| <= d_flip[p], which supplies
    # two more linear inequalities. The sum appears because u is antisymmetric.
    # These distances and d_same are computed once per sample, not once per draw.
    # Shape: (m,). Expected entries: nonnegative floating-point numbers.
    d_flip: np.ndarray

    # Selects the computational strategy: "full" or "cutting-plane".
    # It selects how constraints are supplied, not a different statistic.
    # Both strategies call HiGHS dual simplex through scipy.optimize.linprog.
    # This field is required by the dataclass constructor. The default "full"
    # is supplied by build_lp's function signature, not by this field.
    solver: str

    # Cached full constraint matrix, populated only in full mode.
    # Shape: (q, n), where q is the number of retained pairwise inequalities.
    # Rows already implied by lower/upper are omitted. Each retained row has
    # two nonzero coefficients, so CSR (compressed sparse row) storage avoids
    # allocating a dense q-by-n array. Variable bounds are passed separately.
    #
    # Reading the declaration:
    #   ":"       introduces the type annotation;
    #   "| None"  means that None is also an allowed documented value;
    #   "= None"  is the actual default used when this argument is omitted.
    # None means "no full matrix is stored", not "the LP has no constraints".
    A: csr_matrix | None = None

    # Right-hand-side vector aligned with the rows of A: A @ v <= b.
    # Shape: (q,). Entries are the corresponding d_same or d_flip distances.
    # A and b are filled together. They are both None in cutting-plane mode.
    b: np.ndarray | None = None

    # Constraint IDs used to initialize the cutting-plane working set.
    # A candidate ID is encoded as kind*m+p, with p the pair index and:
    #   kind=0:  v_i-v_j <= d_same[p]
    #   kind=1: -v_i+v_j <= d_same[p]
    #   kind=2:  v_i+v_j <= d_flip[p]
    #   kind=3: -v_i-v_j <= d_flip[p].
    # These IDs refer to the original candidate list, not row numbers in the
    # pruned matrix A. The solver constructs the selected rows from the IDs.
    #
    # None means the first solve has not established a reusable working set.
    # On its first successful solve, solve_lp saves the selected IDs here.
    # In the test workflow that first solve is the observed statistic. Each
    # Monte Carlo solve then starts from a COPY of this same initial set.
    # Extra constraints found during a Monte Carlo solve remain local to it.
    # An empty array is different from None: it records a successful first solve
    # that needed no pairwise constraints. The field is unused in full mode.
    #
    # "Working set" means constraints included in the restricted LP. They need
    # not all be tight (hold with equality) at its optimum, and this cache does
    # not store a simplex basis or a previous primal solution.
    #
    # Shape when populated: (number of cached IDs,). Expected dtype: integer.
    # None also avoids a shared mutable array default. If an array default were
    # needed, dataclasses.field(default_factory=...) would create one per object.
    initial_constraints: np.ndarray | None = None

    # Largest positive feasibility residual among SUCCESSFULLY ACCEPTED solves
    # using this object. It starts at zero for each new sample.
    # solve_lp updates it after checking every original pairwise inequality and
    # every variable bound; negative residuals are treated as zero.
    # The final value therefore covers the observed solve and all MC solves.
    # Intentionally infeasible intermediate cutting-plane solutions are not
    # included in this diagnostic. A failed convergence check raises an error.
    # This is a feasibility diagnostic, not an optimality-gap estimate.
    max_violation: float = 0.0

    # @property makes the method accessible as an attribute: lp.n, not lp.n().
    # It is not a dataclass field and is not an argument to the generated __init__.
    # Computing n from lower avoids storing a second sample-size value that
    # could become inconsistent. No setter is defined, so assigning lp.n fails.
    @property
    def n(self) -> int:
        """Number of LP variables/observations, inferred from the bound vector."""
        return len(self.lower)


def _sample_arrays(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or not x.size:
        raise ValueError("x and y must be nonempty one-dimensional arrays of equal length.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Observations must be finite.")
    return x, y


def _constraint_matrix(lp, ids):
    """Construct either the full matrix or a subset using the same row formulas."""
    ids = np.asarray(ids, dtype=np.int64)
    if not len(ids):
        return coo_matrix((0, lp.n)).tocsr(), np.empty(0)
    kind, pair = np.divmod(ids, len(lp.i))
    columns = np.column_stack([lp.i[pair], lp.j[pair]]).ravel()
    signs = np.array([[1, -1], [-1, 1], [1, 1], [-1, -1]], dtype=float)
    data = signs[kind].ravel()
    matrix = coo_matrix((data, (np.repeat(np.arange(len(ids)), 2), columns)),
                        shape=(len(ids), lp.n)).tocsr()
    rhs = np.where(kind < 2, lp.d_same[pair], lp.d_flip[pair])
    return matrix, rhs


def build_lp(x, y, solver="full"):
    """Build the exact finite LP. Inputs are the coordinates to be tested.

    full: assemble a pruned sparse matrix once and reuse it.
    cutting-plane: cache pairwise distances and generate rows as needed.
    """
    x, y = _sample_arrays(x, y)
    if solver not in ("full", "cutting-plane"):
        raise ValueError("solver must be 'full' or 'cutting-plane'.")
    bound = np.abs(x - y) / np.sqrt(2.0)
    lower = np.where(x < y, -bound, 0.0)
    upper = np.where(x > y, bound, 0.0)
    i, j = np.triu_indices(len(x), k=1)
    same = np.hypot(x[i] - x[j], y[i] - y[j])
    flip = np.hypot(x[i] - y[j], y[i] - x[j])
    lp = LPProblem(lower, upper, i, j, same, flip, solver)
    if solver == "full":
        # Remove only inequalities guaranteed by the variable bounds.
        maxima = (upper[i] - lower[j], upper[j] - lower[i],
                  upper[i] + upper[j], -lower[i] - lower[j])
        rhs = (same, same, flip, flip)
        ids = np.concatenate([k * len(i) + np.flatnonzero(left > right)
                              for k, (left, right) in enumerate(zip(maxima, rhs))])
        lp.A, lp.b = _constraint_matrix(lp, ids)
    return lp


def _solve_linear_program(weights, lp, A, b):
    result = linprog(-weights, A_ub=A, b_ub=b,
                     bounds=np.column_stack([lp.lower, lp.upper]),
                     method="highs-ds",
                     options={"presolve": False,
                              "primal_feasibility_tolerance": FEASIBILITY_TOL,
                              "dual_feasibility_tolerance": FEASIBILITY_TOL})
    if not result.success:
        raise RuntimeError(f"HiGHS failed: {result.message}")
    # The zero vector is feasible, so the true maximum cannot be negative.
    return max(0.0, float(weights @ result.x)), np.asarray(result.x)


def _violations(lp, solution):
    vi, vj = solution[lp.i], solution[lp.j]
    pair = np.concatenate([vi - vj - lp.d_same, vj - vi - lp.d_same,
                           vi + vj - lp.d_flip, -vi - vj - lp.d_flip])
    largest = max(0.0, float(np.max(pair, initial=0.0)),
                  float(np.max(lp.lower - solution, initial=0.0)),
                  float(np.max(solution - lp.upper, initial=0.0)))
    return pair, largest


def solve_lp(weights, lp):
    """Maximize weights @ v over the same SAI feasible set for either solver."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (lp.n,) or not np.isfinite(weights).all():
        raise ValueError(f"weights must be a finite array of length {lp.n}.")
    if lp.solver == "full":
        value, solution = _solve_linear_program(weights, lp, lp.A, lp.b)
        _, largest = _violations(lp, solution)
        if largest > CONSTRAINT_TOL:
            raise RuntimeError(f"Full LP constraint violation {largest:.3g} exceeds tolerance.")
        lp.max_violation = max(lp.max_violation, largest)
        return value

    active = (np.empty(0, dtype=np.int64) if lp.initial_constraints is None
              else lp.initial_constraints.copy())
    included = np.zeros(4 * len(lp.i), dtype=bool)
    included[active] = True
    # Each unsuccessful round adds a new original constraint. There are finitely many.
    for _ in range(len(included) + 1):
        A, b = _constraint_matrix(lp, active)
        value, solution = _solve_linear_program(weights, lp, A, b)
        violations, largest = _violations(lp, solution)
        if largest <= CONSTRAINT_TOL:
            lp.max_violation = max(lp.max_violation, largest)
            if lp.initial_constraints is None:
                lp.initial_constraints = active.copy()
            return value
        candidates = np.flatnonzero((violations > CONSTRAINT_TOL) & ~included)
        if not len(candidates):
            raise RuntimeError("Existing LP constraints exceed tolerance; convergence is not certified.")
        if len(candidates) > CUT_BATCH_SIZE:
            chosen = np.argpartition(violations[candidates], -CUT_BATCH_SIZE)[-CUT_BATCH_SIZE:]
            candidates = candidates[chosen]
        candidates = np.sort(candidates)
        included[candidates] = True
        active = np.concatenate([active, candidates])
    raise RuntimeError("Constraint generation did not converge.")


def monte_carlo_lipsai_test(x, y, B=199, alpha=0.05, rng=None,
                           solver="full", transform=True):
    """Construct the observed statistic, B calibration values, and plus-one p-value.

    With transform=True, apply the same logistic map to both coordinates.
    Set transform=False for observations already on the desired bounded scale.
    """
    x, y = _sample_arrays(x, y)
    if not isinstance(B, (int, np.integer)) or isinstance(B, bool) or B < 1:
        raise ValueError("B must be a positive integer.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")
    rng = np.random.default_rng() if rng is None else rng
    if transform:
        x, y = expit(x), expit(y)
    lp = build_lp(x, y, solver=solver)
    signs = 2.0 * rng.integers(0, 2, size=(B, lp.n), dtype=np.int8).astype(float) - 1.0
    statistic = solve_lp(np.full(lp.n, 2.0 / lp.n), lp)
    mc_values = np.array([solve_lp((2.0 / lp.n) * sign, lp) for sign in signs])
    p_value = float((1 + np.count_nonzero(mc_values >= statistic)) / (B + 1))
    # Use the exact rank equivalent to the plus-one rule, including small B.
    m = int(np.count_nonzero(np.arange(1, B + 2) / (B + 1) <= alpha))
    critical = float(np.partition(mc_values, B - m)[B - m]) if m else float("inf")
    return {"statistic": statistic, "critical_value": critical,
            "p_value": p_value, "reject": p_value <= alpha,
            "mc_values": mc_values, "max_violation": lp.max_violation}
