import functools
from typing import Callable, Optional

import numpy as np

from ._flooring import identity, max_flooring

EPS = 1e-10


def to_psd(
    X: np.ndarray,
    axis1: int = -2,
    axis2: int = -1,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
) -> np.ndarray:
    r"""Ensure matrix to be positive semidefinite.

    Args:
        X (np.ndarray):
            A complex Hermitian matrix.
        axis1 (int):
            Axis to be used as first axis of 2D sub-arrays.
        axis2 (int):
            Axis to be used as second axis of 2D sub-arrays.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.

    Returns:
        Positive semidefinite matrix.
    """
    if flooring_fn is None:
        flooring_fn = identity

    shape = X.shape
    n_dims = len(shape)

    axis1 = n_dims + axis1 if axis1 < 0 else axis1
    axis2 = n_dims + axis2 if axis2 < 0 else axis2

    assert axis1 == n_dims - 2 and axis2 == n_dims - 1, "axis1 == -2 and axis2 == -1"

    if np.iscomplexobj(X):
        X = (X + X.swapaxes(axis1, axis2).conj()) / 2
    else:
        X = (X + X.swapaxes(axis1, axis2)) / 2

    eigvals = np.linalg.eigvalsh(X)
    delta = np.min(eigvals, axis=-1)
    delta = np.minimum(delta, 0)  # 0 is expected if possible
    trace = np.trace(X, axis1=axis1, axis2=axis2).real

    zeros = np.zeros(())
    eye = np.eye(shape[-1])

    eps = np.maximum(flooring_fn(zeros) * trace - delta, 0)
    X = X + eps[..., np.newaxis, np.newaxis] * eye

    return X
