from typing import Callable, Optional

import numpy as np

from .eigh import eigh


def sqrtmh(X: np.ndarray) -> np.ndarray:
    r"""Compute square root of a positive semidefinite Hermitian or symmetric matrix.

    Args:
        X (numpy.ndarray):
            A complex Hermitian or symmetric matrix with shape of (\*, n_channels, n_channels).

    Returns:
        numpy.ndarray of square root. The shape is same as that of input.
    """
    Lamb, P = eigh(X)

    P_Hermite = P.swapaxes(-2, -1)

    if np.iscomplexobj(X):
        P_Hermite = P_Hermite.conj()

    Lamb = np.sqrt(Lamb)[..., np.newaxis] * np.eye(Lamb.shape[-1])

    return P @ Lamb @ P_Hermite


def invsqrtmh(
    X: np.ndarray,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    r"""Compute inversion of square root for a positive definite Hermitian or symmetric matrix.

    Args:
        X (numpy.ndarray):
            A complex Hermitian matrix with shape of (\*, n_channels, n_channels).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to receive and return the same shape as that of X.
            By default, the identity function (``lambda x: x``) is used.

    Returns:
        numpy.ndarray of inversion of square root. The shape is same as that of input.
    """

    def _identity(x):
        return x

    if flooring_fn is None:
        flooring_fn = _identity

    Lamb, P = eigh(X)

    P_Hermite = P.swapaxes(-2, -1)

    if np.iscomplexobj(X):
        P_Hermite = P_Hermite.conj()

    Lamb = 1 / flooring_fn(np.sqrt(Lamb))
    Lamb = Lamb[..., np.newaxis] * np.eye(Lamb.shape[-1])

    return P @ Lamb @ P_Hermite
