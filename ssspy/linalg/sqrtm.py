import numpy as np

from .eigh import eigh


def sqrtmh(X: np.ndarray) -> np.ndarray:
    r"""Compute square root of a complex Hermitian (conjugate symmetric) or a real symmetric matrix.

    Args:
        X (numpy.ndarray):
            A complex Hermitian matrix with shape of (\*, n_channels, n_channels).

    Returns:
        numpy.ndarray of square root. The shape is same as that of input.
    """
    Lamb, P = eigh(X)

    P_Hermite = P.swapaxes(-2, -1)

    if np.iscomplexobj(X):
        P_Hermite = P_Hermite.conj()

    Lamb = np.sqrt(Lamb)[..., np.newaxis] * np.eye(Lamb.shape[-1])

    return P @ Lamb @ P_Hermite
