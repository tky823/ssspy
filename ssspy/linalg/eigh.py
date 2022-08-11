from typing import Optional, Tuple, Union

import numpy as np

from .inv import inv2


def eigh(
    A: np.ndarray, B: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""Generalized eigenvalue decomposition.

    Solve :math:`\boldsymbol{A}\boldsymbol{z} = \lambda\boldsymbol{B}\boldsymbol{z}`, \
    and return :math:`\boldsymbol{z}`.

    Args:
        A (numpy.ndarray):
            A complex Hermitian matrix with shape of (\*, n_channels, n_channels).
        B (numpy.ndarray, optional):
            A complex Hermitian matrix with shape of (\*, n_channels, n_channels).

    Returns:
        A tuple of (eigenvalues, eigenvectors)
            - Eigenvalues have shape of (\*, n_channels).
            - Eigenvectors have shape of (\*, n_channels, n_channels).
    """
    if B is None:
        return np.linalg.eigh(A)

    L = np.linalg.cholesky(B)
    L_inv = np.linalg.inv(L)

    L_inv_Hermite = np.swapaxes(L_inv, -2, -1)

    if np.iscomplexobj(L_inv_Hermite):
        L_inv_Hermite = L_inv_Hermite.conj()

    C = L_inv @ A @ L_inv_Hermite
    lamb, y = np.linalg.eigh(C)
    z = L_inv_Hermite @ y

    return lamb, z


def eigh2(
    A: np.ndarray, B: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""Generalized eigenvalue decomposition for 2x2 matrix.

    Solve :math:`\boldsymbol{A}\boldsymbol{z} = \lambda\boldsymbol{B}\boldsymbol{z}`, \
    and return :math:`\boldsymbol{z}`.
    See also https://github.com/tky823/ssspy/issues/115 for this implementation.

    Args:
        A (numpy.ndarray):
            A complex Hermitian matrix with shape of (\*, 2, 2).
        B (numpy.ndarray, optional):
            A complex Hermitian matrix with shape of (\*, 2, 2).

    Returns:
        A tuple of (eigenvalues, eigenvectors)
            - Eigenvalues have shape of (\*, 2).
            - Eigenvectors have shape of (\*, 2, 2).
    """
    assert A.shape[-2:] == (2, 2), "2x2 matrix is expected, but given shape of {}.".format(A.shape)

    if B is None:
        return np.linalg.eigh(A)

    L = np.linalg.cholesky(B)
    L_inv = inv2(L)
    L_inv_Hermite = np.swapaxes(L_inv, -2, -1)

    if np.iscomplexobj(L_inv_Hermite):
        L_inv_Hermite = L_inv_Hermite.conj()

    C = L_inv @ A @ L_inv_Hermite
    lamb, y = eigh2(C)
    z = L_inv_Hermite @ y

    return lamb, z
