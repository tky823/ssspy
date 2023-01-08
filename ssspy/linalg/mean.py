import numpy as np

from .sqrtm import invsqrtmh, sqrtmh


def gmeanmh(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""Compute the geometric mean of complex Hermitian \
    (conjugate symmetric) or real symmetric matrices.

    .. math::
        \boldsymbol{A}\#\boldsymbol{B}
        \boldsymbol{A}^{1/2}
        (\boldsymbol{A}^{-1/2}B\boldsymbol{A}^{-1/2})^{1/2}
        \boldsymbol{A}^{1/2}

    Args:
        A (numpy.ndarray):
            A complex Hermitian matrix with shape of (\*, n_channels, n_channels).
        B (numpy.ndarray, optional):
            A complex Hermitian matrix with shape of (\*, n_channels, n_channels).

    Returns:
        Geometric mean of matrices with shape of (\*, n_channels, n_channels).
    """  # noqa: W605
    A_sqrt = sqrtmh(A)
    A_invsqrtmh = invsqrtmh(A)

    X = A_sqrt @ sqrtmh(A_invsqrtmh @ B @ A_invsqrtmh) @ A_sqrt

    return X
