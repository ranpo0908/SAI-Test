"""Sample generators and the distribution parameters used in the manuscript."""

import numpy as np


CASE_PARAMETERS = {
    "gaussian_null": {"family": "normal", "location": [0, 1],
                      "scale": [[1, 0.3], [0.3, 1]]},
    "gaussian_alt": {"family": "normal", "location": [2, 1],
                     "scale": [[1, 0.5], [0.5, 2]]},
    "gaussian_near": {"family": "normal", "location": [1, 1],
                      "scale": [[4, 0.4], [0.4, 1]]},
    "skew_null": {"family": "skew-normal", "location": [0, 1],
                  "scale": [[1, 0.3], [0.3, 4]], "shape": [1, 4]},
    "skew_alt": {"family": "skew-normal", "location": [1, 1],
                 "scale": [[1, 0.3], [0.3, 4]], "shape": [4, 1]},
    "skew_near": {"family": "skew-normal", "location": [0, 1],
                  "scale": [[1, 0.3], [0.3, 1]], "shape": [4, 1]},
}


def _parameters(n, location, scale):
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer.")
    location, scale = np.asarray(location, dtype=float), np.asarray(scale, dtype=float)
    if location.shape != (2,) or scale.shape != (2, 2):
        raise ValueError("location must have length two and scale must be a 2-by-2 matrix.")
    if not np.isfinite(location).all() or not np.isfinite(scale).all():
        raise ValueError("Distribution parameters must be finite.")
    if not np.allclose(scale, scale.T, rtol=0, atol=1e-12):
        raise ValueError("scale must be symmetric.")
    np.linalg.cholesky(scale)
    return location, scale


def sample_bivariate_normal(n, rng, location, scale):
    """For a normal distribution, scale is the covariance matrix."""
    location, scale = _parameters(n, location, scale)
    sample = rng.multivariate_normal(location, scale, size=n)
    return sample[:, 0], sample[:, 1]


def sample_bivariate_skew_normal(n, rng, location, scale, shape):
    """Azzalini-type SN(location, scale, shape), as in the original repository.

    scale is the skew-normal scale matrix, not the resulting covariance matrix.
    """
    location, scale = _parameters(n, location, scale)
    shape = np.asarray(shape, dtype=float)
    if shape.shape != (2,) or not np.isfinite(shape).all():
        raise ValueError("shape must contain two finite entries.")
    omega = np.sqrt(np.diag(scale))
    correlation = scale / np.outer(omega, omega)
    correlation = (correlation + correlation.T) / 2
    delta = correlation @ shape / np.sqrt(1 + shape @ correlation @ shape)
    residual = correlation - np.outer(delta, delta)
    eigenvalues, eigenvectors = np.linalg.eigh((residual + residual.T) / 2)
    if eigenvalues.min() < -1e-10:
        raise ValueError("Invalid skew-normal residual covariance.")
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    u0 = np.abs(rng.normal(size=n))[:, None]
    u1 = rng.normal(size=(n, 2))
    sample = location + (u0 * delta + u1 @ root.T) * omega
    return sample[:, 0], sample[:, 1]


def case_parameters(case, a=None):
    """Return explicit parameters; used for sampling and recording run settings."""
    if case == "gaussian_path":
        if a is None or not np.isfinite(a):
            raise ValueError("The Gaussian path requires a finite a.")
        return {"family": "normal", "location": [0, float(a)],
                "scale": [[1, 0.3], [0.3, 1]]}
    if case == "skew_path":
        if a is None or not np.isfinite(a):
            raise ValueError("The skew-normal path requires a finite a.")
        return {"family": "skew-normal", "location": [0, float(a)],
                "scale": [[1, 0.3], [0.3, 4]], "shape": [1, 4]}
    if case not in CASE_PARAMETERS:
        raise ValueError(f"Unknown distribution case: {case}")
    # Return fresh lists so callers cannot mutate the registry accidentally.
    parameters = CASE_PARAMETERS[case]
    return {key: [row.copy() for row in value] if key == "scale"
            else value.copy() if isinstance(value, list) else value
            for key, value in parameters.items()}


def sample_case(case, n, rng, a=None):
    parameters = case_parameters(case, a)
    if parameters["family"] == "normal":
        return sample_bivariate_normal(n, rng, parameters["location"], parameters["scale"])
    return sample_bivariate_skew_normal(n, rng, parameters["location"],
                                       parameters["scale"], parameters["shape"])
