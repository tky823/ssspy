import numpy as np

from .eigh import eigh


def gmeanmh(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""Compute the geometric mean of complex Hermitian \
    (conjugate symmetric) or real symmetric matrices.

    .. math::
        \boldsymbol{A}\#\boldsymbol{B}
        &= \boldsymbol{A}^{1/2}
        (\boldsymbol{A}^{-1/2}B\boldsymbol{A}^{-1/2})^{1/2}
        \boldsymbol{A}^{1/2} \\
        &= \boldsymbol{A}(\boldsymbol{A}^{-1}\boldsymbol{B})^{1/2} \\
        &= (\boldsymbol{A}\boldsymbol{B}^{-1})^{1/2}\boldsymbol{B}

    .. note::
        :math:`(\boldsymbol{A}^{-1}\boldsymbol{B})^{1/2}` is computed by
        generalized eigendecomposition.
        Let :math:`\lambda` and :math:`z` be the eigenvalue and eigenvector of
        the generalized eigenproblem :math:`\boldsymbol{Bz}=\lambda\boldsymbol{Az}`.
        Then, :math:`(\boldsymbol{A}^{-1}\boldsymbol{B})^{1/2}` is computed by
        :math:`\boldsymbol{Z}\boldsymbol{\Lambda}^{1/2}\boldsymbol{Z}^{-1}`,
        where the main diagonals of :math:`\boldsymbol{\Lambda}` are :math:`\lambda` s
        and the columns of :math:`\boldsymbol{Z}` are :math:`\boldsymbol{z}` s.

    Args:
        A (numpy.ndarray):
            A complex Hermitian matrix with shape of (\*, n_channels, n_channels).
        B (numpy.ndarray, optional):
            A complex Hermitian matrix with shape of (\*, n_channels, n_channels).

    Returns:
        Geometric mean of matrices with shape of (\*, n_channels, n_channels).
    """  # noqa: W605
    lamb, Z = eigh(B, A)
    lamb = np.sqrt(lamb)
    Lamb = lamb[..., np.newaxis] * np.eye(Z.shape[-1])
    AB = Z @ Lamb @ np.linalg.inv(Z)

    return A @ AB
