import numpy as np

from .inv import inv2


def eigh(A: np.ndarray, B: np.ndarray = None) -> np.ndarray:
    r"""Generalized eigenvalue decomposition.

    Args:
        A (numpy.ndarray):
            A complex Hermitian matrix with shape of (*, n_channels, n_channels).
        B (numpy.ndarray, optional):
            A complex Hermitian matrix with shape of (*, n_channels, n_channels).

    Returns:
        numpy.ndarray:
            Eigenvalues with shape of (*, n_channels).
        numpy.ndarray:
            Eigenvectors with shape of (*, n_channels, n_channels).

    Solve :math:`\boldsymbol{A}\boldsymbol{z} = \lambda\boldsymbol{B}\boldsymbol{z}`, \
    and return :math:`\boldsymbol{z}`.
    """
    if B is None:
        return np.linalg.eigh(A)

    L = np.linalg.cholesky(B)
    L_inv = np.linalg.inv(L)
    L_inv_Hermite = np.swapaxes(L_inv, -2, -1).conj()
    C = L_inv @ A @ L_inv_Hermite
    lamb, y = np.linalg.eigh(C)
    z = L_inv_Hermite @ y

    return lamb, z


def eigh2(A: np.ndarray, B: np.ndarray = None) -> np.ndarray:
    r"""Generalized eigenvalue decomposition for 2x2 matrix.

    Args:
        A (numpy.ndarray):
            A complex Hermitian matrix with shape of (*, 2, 2).
        B (numpy.ndarray, optional):
            A complex Hermitian matrix with shape of (*, 2, 2).

    Returns:
        numpy.ndarray:
            Eigenvalues with shape of (*, 2).
        numpy.ndarray:
            Eigenvectors with shape of (*, 2, 2).

    Solve :math:`\boldsymbol{A}\boldsymbol{z} = \lambda\boldsymbol{B}\boldsymbol{z}`, \
    and return :math:`\boldsymbol{z}`.
    """
    assert A.shape[-2:] == (2, 2), "2x2 matrix is expected, but given shape of {}.".format(A.shape)

    if B is None:
        a = A[..., 0, 0]
        b = A[..., 0, 1]
        c = A[..., 1, 0]
        d = A[..., 1, 1]

        if np.iscomplexobj(A):
            a = np.real(a)
            d = np.real(d)

            assert np.allclose(c.conj(), b), "Matrix should be positive semidefinite."
        else:
            assert np.allclose(c, b), "Matrix should be positive semidefinite."

        trace = a + d
        det = a * d - np.abs(b) ** 2
        sqrt_discriminant = np.sqrt(trace ** 2 - 4 * det)
        lamb_pos = trace + sqrt_discriminant
        lamb_neg = trace - sqrt_discriminant
        v_00 = (d - a - sqrt_discriminant) / 2
        v_pos = np.stack([v_00, -c], axis=-1)
        v_neg = np.stack([-b, -v_00], axis=-1)

        lamb = np.stack([lamb_neg, lamb_pos], axis=-1) / 2
        v = np.stack([v_neg, v_pos], axis=-1)

        return lamb, v

    L = np.linalg.cholesky(B)
    L_inv = inv2(L)
    L_inv_Hermite = np.swapaxes(L_inv, -2, -1).conj()
    C = L_inv @ A @ L_inv_Hermite
    lamb, y = eigh2(C)
    z = L_inv_Hermite @ y

    return lamb, z
