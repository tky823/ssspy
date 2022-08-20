from typing import Optional, Tuple, Union

import numpy as np

from .inv import inv2


def eigh(
    A: np.ndarray, B: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""Compute the (generalized) eigenvalues and eigenvectors of a complex Hermitian \
    (conjugate symmetric) or a real symmetric matrix.

    If ``B`` is ``None``, solve :math:`\boldsymbol{A}\boldsymbol{z} = \lambda\boldsymbol{z}`.

    If ``B`` is given, \
    solve :math:`\boldsymbol{A}\boldsymbol{z} = \lambda\boldsymbol{B}\boldsymbol{z}`.

    Args:
        A (numpy.ndarray):
            A complex Hermitian matrix with shape of (\*, n_channels, n_channels).
        B (numpy.ndarray, optional):
            A complex Hermitian matrix with shape of (\*, n_channels, n_channels).

    Returns:
        A tuple of (eigenvalues, eigenvectors)
            - Eigenvalues have shape of (\*, n_channels).
            - Eigenvectors have shape of (\*, n_channels, n_channels).

    .. note::
        If ``B`` is given, we use cholesky decomposition to \
        satisfy :math:`\boldsymbol{L}\boldsymbol{L}^{\mathsf{H}}=\boldsymbol{B}`.

        Then, solve :math:`\boldsymbol{C}\boldsymbol{y} = \lambda\boldsymbol{y}`, \
        where :math:`\boldsymbol{C}=\boldsymbol{L}^{-1}\boldsymbol{A}\boldsymbol{L}^{-\mathsf{H}}`.

        The generalized eigenvalues of :math:`\boldsymbol{A}` and :math:`\boldsymbol{B}` \
        are computed by :math:`\boldsymbol{L}^{-\mathsf{H}}\boldsymbol{y}`.

    Examples::

        .. code-block:: python

            >>> import numpy as np
            >>> from ssspy.linalg import eigh
            >>> A = np.array([[1, -2j], [2j, 3]])
            >>> lamb, z = eigh(A)
            >>> lamb; z
            array([-0.23606798,  4.23606798])
            array([[-0.85065081+0.j        , -0.52573111+0.j        ],
                [ 0.        +0.52573111j,  0.        -0.85065081j]])
            >>> np.allclose(A @ z, lamb * z)
            True

        .. code-block:: python

            >>> import numpy as np
            >>> from ssspy.linalg import eigh
            >>> A = np.array([[1, -2j], [2j, 3]])
            >>> B = np.array([[2, -3j], [3j, 5]])
            >>> lamb, z = eigh(A, B)
            >>> lamb; z
            array([-1.61803399,  0.61803399])
            array([[ 2.22703273+0.j        , -0.20081142+0.j        ],
                [ 0.        -1.37638192j,  0.        -0.3249197j ]])
            >>> np.allclose(A @ z, lamb * (B @ z))
            True
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
    r"""Compute the (generalized) eigenvalues and eigenvectors of a 2x2 complex Hermitian \
    (conjugate symmetric) or a real symmetric matrix.

    If ``B`` is ``None``, solve :math:`\boldsymbol{A}\boldsymbol{z} = \lambda\boldsymbol{z}`.

    If ``B`` is given, \
    solve :math:`\boldsymbol{A}\boldsymbol{z} = \lambda\boldsymbol{B}\boldsymbol{z}`.

    Args:
        A (numpy.ndarray):
            A complex Hermitian matrix with shape of (\*, 2, 2).
        B (numpy.ndarray, optional):
            A complex Hermitian matrix with shape of (\*, 2, 2).

    Returns:
        A tuple of (eigenvalues, eigenvectors)
            - Eigenvalues have shape of (\*, 2).
            - Eigenvectors have shape of (\*, 2, 2).

    .. note::
        If ``B`` is given, we use cholesky decomposition to \
        satisfy :math:`\boldsymbol{L}\boldsymbol{L}^{\mathsf{H}}=\boldsymbol{B}`.

        Then, solve :math:`\boldsymbol{C}\boldsymbol{y} = \lambda\boldsymbol{y}`, \
        where :math:`\boldsymbol{C}=\boldsymbol{L}^{-1}\boldsymbol{A}\boldsymbol{L}^{-\mathsf{H}}`.

        The generalized eigenvalues of :math:`\boldsymbol{A}` and :math:`\boldsymbol{B}` \
        are computed by :math:`\boldsymbol{L}^{-\mathsf{H}}\boldsymbol{y}`.

        See also https://github.com/tky823/ssspy/issues/115 for this implementation.

    Examples::

        .. code-block:: python

            >>> import numpy as np
            >>> from ssspy.linalg import eigh2
            >>> A = np.array([[1, -2j], [2j, 3]])
            >>> lamb, z = eigh2(A)
            >>> lamb; z
            array([-0.23606798,  4.23606798])
            array([[-0.85065081+0.j        , -0.52573111+0.j        ],
                [ 0.        +0.52573111j,  0.        -0.85065081j]])
            >>> np.allclose(A @ z, lamb * z)
            True

        .. code-block:: python

            >>> import numpy as np
            >>> from ssspy.linalg import eigh2
            >>> A = np.array([[1, -2j], [2j, 3]])
            >>> B = np.array([[2, -3j], [3j, 5]])
            >>> lamb, z = eigh2(A, B)
            >>> lamb; z
            array([-1.61803399,  0.61803399])
            array([[ 2.22703273+0.j        , -0.20081142+0.j        ],
                [ 0.        -1.37638192j,  0.        -0.3249197j ]])
            >>> np.allclose(A @ z, lamb * (B @ z))
            True
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
