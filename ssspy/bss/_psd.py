import functools
import warnings
from typing import Callable, Optional

import numpy as np

from ..special.flooring import max_flooring
from ..special.psd import to_psd as _to_psd

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
    warnings.warn("Use ssspy.special.to_psd instead.", FutureWarning)

    return _to_psd(X, axis1=axis1, axis2=axis2, flooring_fn=flooring_fn)
