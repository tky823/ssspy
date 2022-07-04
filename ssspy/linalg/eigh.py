import numpy as np


def eigh(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""Generalized eigendecomposition.

    Args:
        A (numpy.ndarray):
            A complex Hermitian matrix with shape of (*, n_channels, n_channels).
        B (numpy.ndarray):
            A complex Hermitian matrix with shape of (*, n_channels, n_channels).

    Returns:
        numpy.ndarray:
            Eigenvalues with shape of (*, n_channels).
        numpy.ndarray:
            Eigenvectors with shape of (*, n_channels, n_channels).

    Solve :math:`\boldsymbol{A}\boldsymbol{z} = \lambda\boldsymbol{B}\boldsymbol{z}`, \
    and return :math:`\boldsymbol{z}`.
    """
    L = np.linalg.cholesky(B)
    L_inv = np.linalg.inv(L)
    L_inv_Hermite = np.swapaxes(L_inv, -2, -1).conj()
    C = L_inv @ A @ L_inv_Hermite
    lamb, y = np.linalg.eigh(C)
    z = L_inv_Hermite @ y

    return lamb, z
