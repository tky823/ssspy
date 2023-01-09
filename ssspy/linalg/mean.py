import numpy as np

from .eigh import eigh


def gmeanmh(A: np.ndarray, B: np.ndarray, type: int = 1) -> np.ndarray:
    r"""Compute the geometric mean of complex Hermitian \
    (conjugate symmetric) or real symmetric matrices.

    The geometric mean of :math:`\boldsymbol{A}` and :math:`\boldsymbol{B}`
    is defined as follows [#bhatia2009positive]_:

    .. math::
        \boldsymbol{A}\#\boldsymbol{B}
        &= \boldsymbol{A}^{1/2}
        (\boldsymbol{A}^{-1/2}\boldsymbol{B}\boldsymbol{A}^{-1/2})^{1/2}
        \boldsymbol{A}^{1/2} \\
        &= \boldsymbol{A}(\boldsymbol{A}^{-1}\boldsymbol{B})^{1/2} \\
        &= (\boldsymbol{A}\boldsymbol{B}^{-1})^{1/2}\boldsymbol{B}.

    This is a solution of the following equation for
    complex Hermitian or real symmetric matrices,
    :math:`\boldsymbol{A}`, :math:`\boldsymbol{B}`, and :math:`\boldsymbol{X}`:

    .. math::
        \boldsymbol{X}\boldsymbol{A}^{-1}\boldsymbol{X} = \boldsymbol{B}.

    .. note::
        In this toolkit, :math:`\boldsymbol{A}\#\boldsymbol{B}` is computed by
        :math:`\boldsymbol{B}(\boldsymbol{B}^{-1}\boldsymbol{A})^{1/2}`
        in terms of computational speed.
        Note that :math:`\boldsymbol{A}\#\boldsymbol{B}` is equal to
        :math:`\boldsymbol{B}\#\boldsymbol{A}`.
        For comparison of computational time, see https://github.com/tky823/ssspy/issues/210.

    .. note::
        :math:`(\boldsymbol{B}^{-1}\boldsymbol{A})^{1/2}` is computed by
        generalized eigendecomposition.
        Let :math:`\lambda` and :math:`z` be the eigenvalue and eigenvector of
        the generalized eigenproblem :math:`\boldsymbol{Az}=\lambda\boldsymbol{Bz}`.
        Then, :math:`(\boldsymbol{B}^{-1}\boldsymbol{A})^{1/2}` is computed by
        :math:`\boldsymbol{Z}\boldsymbol{\Lambda}^{1/2}\boldsymbol{Z}^{-1}`,
        where the main diagonals of :math:`\boldsymbol{\Lambda}` are :math:`\lambda` s
        and the columns of :math:`\boldsymbol{Z}` are :math:`\boldsymbol{z}` s.

    Args:
        A (numpy.ndarray):
            A complex Hermitian matrix with shape of (\*, n_channels, n_channels).
        B (numpy.ndarray):
            A complex Hermitian matrix with shape of (\*, n_channels, n_channels).
        type (int):
            This value specifies the type of geometric mean.
            Only ``1``, ``2``, and ``3`` are supported.

            - When ``type=1``, return :math:`\boldsymbol{A}\#\boldsymbol{B}`.
            - When ``type=2``, return :math:`\boldsymbol{A}^{-1}\#\boldsymbol{B}`.
            - When ``type=3``, return :math:`\boldsymbol{A}\#\boldsymbol{B}^{-1}`.

    Returns:
        Geometric mean of matrices with shape of (\*, n_channels, n_channels).

    .. [#bhatia2009positive] R. Bhatia,
        "Positive definite matrices,"
        Princeton university press, 2009.
    """  # noqa: W605
    lamb, Z = eigh(A, B, type=type)
    lamb = np.sqrt(lamb)
    Lamb = lamb[..., np.newaxis] * np.eye(Z.shape[-1])
    ZLZ = Z @ Lamb @ np.linalg.inv(Z)

    if type == 1:
        BA = ZLZ
        G = B @ BA
    elif type == 2:
        AB = ZLZ
        G = np.linalg.inv(A) @ AB
    elif type == 3:
        BA = ZLZ
        G = np.linalg.inv(B) @ BA
    else:
        raise ValueError("Invalid type={} is given.".format(type))

    return G
