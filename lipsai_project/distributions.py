from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    family: str
    truth: str
    description: str
    sampler: Callable[[int, np.random.Generator], Tuple[np.ndarray, np.ndarray]]



def _make_bvn_sampler(mean: np.ndarray, cov: np.ndarray):
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    def sampler(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        sample = rng.multivariate_normal(mean=mean, cov=cov, size=n)
        return sample[:, 0], sample[:, 1]

    return sampler



def sample_bivariate_skew_normal(
    n: int,
    rng: np.random.Generator,
    *,
    location: np.ndarray,
    scale: np.ndarray,
    shape: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate samples from the Azzalini-type multivariate skew-normal SN_d(xi, Omega, alpha).

    Here `location` is xi, `scale` is Omega, and `shape` is alpha.
    The `scale` matrix is the skew-normal scale matrix, not the final covariance matrix.
    """
    location = np.asarray(location, dtype=float)
    scale = np.asarray(scale, dtype=float)
    shape = np.asarray(shape, dtype=float)

    if scale.shape != (2, 2):
        raise ValueError('This helper currently supports only the bivariate case.')
    if location.shape != (2,) or shape.shape != (2,):
        raise ValueError('location and shape must be length-2 arrays.')

    omega = np.sqrt(np.diag(scale))
    if np.any(omega <= 0):
        raise ValueError('The diagonal of the scale matrix must be strictly positive.')

    omega_inv = np.diag(1.0 / omega)
    corr = omega_inv @ scale @ omega_inv
    corr = 0.5 * (corr + corr.T)

    delta = corr @ shape
    delta = delta / np.sqrt(1.0 + shape @ corr @ shape)

    residual = corr - np.outer(delta, delta)
    residual = 0.5 * (residual + residual.T)
    eigvals, eigvecs = np.linalg.eigh(residual)
    if np.min(eigvals) < -1e-10:
        raise ValueError('The implied residual matrix is not positive semidefinite.')
    eigvals = np.clip(eigvals, 0.0, None)
    residual_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

    u0 = np.abs(rng.normal(size=n))[:, None]
    u1 = rng.normal(size=(n, 2))
    z = u0 * delta + u1 @ residual_sqrt.T
    sample = location + z * omega
    return sample[:, 0], sample[:, 1]



def _make_bvsn_sampler(location: np.ndarray, scale: np.ndarray, shape: np.ndarray):
    location = np.asarray(location, dtype=float)
    scale = np.asarray(scale, dtype=float)
    shape = np.asarray(shape, dtype=float)

    def sampler(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        return sample_bivariate_skew_normal(
            n=n,
            rng=rng,
            location=location,
            scale=scale,
            shape=shape,
        )

    return sampler



def make_skew_power_case(a: float) -> ExperimentCase:
    """Generated case for the skew-normal location sweep.

    Family: SN([0,a], [[1,0.3],[0.3,4]], [1,4]).
    """

    a_float = float(a)
    name_token = f'{a_float:+.3f}'.replace('+', 'p').replace('-', 'm').replace('.', 'p')
    return ExperimentCase(
        name=f'skew_power_a_{name_token}',
        family='bivariate_skew_normal',
        truth='unknown',
        description=f'BivSkewNorm(loc=[0,{a_float:g}], scale=[[1,0.3],[0.3,4]], shape=[1,4])',
        sampler=_make_bvsn_sampler(
            location=np.array([0.0, a_float]),
            scale=np.array([[1.0, 0.3], [0.3, 4.0]]),
            shape=np.array([1.0, 4.0]),
        ),
    )


CASE_REGISTRY: Dict[str, ExperimentCase] = {
    'null_l_sai': ExperimentCase(
        name='null_l_sai',
        family='bivariate_normal',
        truth='L-SAI',
        description='BivNorm([0,1], [[1,0.3],[0.3,1]])',
        sampler=_make_bvn_sampler(
            mean=np.array([0.0, 1.0]),
            cov=np.array([[1.0, 0.3], [0.3, 1.0]]),
        ),
    ),
    'alt_mean_cov': ExperimentCase(
        name='alt_mean_cov',
        family='bivariate_normal',
        truth='not L-SAI',
        description='BivNorm([2,1], [[1,0.5],[0.5,2]])',
        sampler=_make_bvn_sampler(
            mean=np.array([2.0, 1.0]),
            cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        ),
    ),
    'alt_var': ExperimentCase(
        name='alt_var',
        family='bivariate_normal',
        truth='not L-SAI',
        description='BivNorm([1,1], [[4,0.4],[0.4,1]])',
        sampler=_make_bvn_sampler(
            mean=np.array([1.0, 1.0]),
            cov=np.array([[4.0, 0.4], [0.4, 1.0]]),
        ),
    ),
    'skew_14': ExperimentCase(
        name='skew_14',
        family='bivariate_skew_normal',
        truth='unknown',
        description='BivSkewNorm(loc=[0,1], scale=[[1,0.3],[0.3,4]], shape=[1,4])',
        sampler=_make_bvsn_sampler(
            location=np.array([0.0, 1.0]),
            scale=np.array([[1.0, 0.3], [0.3, 4.0]]),
            shape=np.array([1.0, 4.0]),
        ),
    ),
    'skew_41': ExperimentCase(
        name='skew_41',
        family='bivariate_skew_normal',
        truth='unknown',
        description='BivSkewNorm(loc=[0,1], scale=[[1,0.3],[0.3,4]], shape=[4,1])',
        sampler=_make_bvsn_sampler(
            location=np.array([0.0, 1.0]),
            scale=np.array([[1.0, 0.3], [0.3, 4.0]]),
            shape=np.array([4.0, 1.0]),
        ),
    ),
    'skew_11': ExperimentCase(
        name='skew_11',
        family='bivariate_skew_normal',
        truth='unknown',
        description='BivSkewNorm(loc=[1,1], scale=[[1,0.3],[0.3,4]], shape=[4,1])',
        sampler=_make_bvsn_sampler(
            location=np.array([0.0, 1.0]),
            scale=np.array([[1.0, 0.3], [0.3, 4.0]]),
            shape=np.array([4.0, 1.0]),
        ),
    ),
}



def list_case_names() -> list[str]:
    return list(CASE_REGISTRY.keys())
