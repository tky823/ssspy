import numpy as np


def inv2(X: np.ndarray) -> np.ndarray:
    r"""Compute the (multiplicative) inverse of a 2x2 matrix.

    Args:
        X (numpy.ndarray):
            2x2 matrix to be inverted. The shape is (\*, 2, 2).

    Returns:
        numpy.ndarray:
            (Multiplicative) inverse of the matrix X.

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from ssspy.linalg import inv2
            >>> X = np.array([[0, 1], [2, 3]])
            >>> X_inv = inv2(X)
            >>> np.allclose(X @ X_inv, np.eye(2))
            True
            >>> np.allclose(X_inv @ X, np.eye(2))
            True

        .. code-block:: python

            >>> import numpy as np
            >>> from ssspy.linalg import inv2
            >>> X = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
            >>> inv2(X)
            array([[[-1.5,  0.5],
                    [ 1. , -0. ]],

                [[-3.5,  2.5],
                    [ 3. , -2. ]]])
    """
    shape = X.shape

    assert shape[-2:] == (2, 2), "2x2 matrix is expected, but given shape of {}.".format(shape)

    a = X[..., 0, 0]
    b = X[..., 0, 1]
    c = X[..., 1, 0]
    d = X[..., 1, 1]

    det = a * d - b * c

    X_adj = np.stack([d, -b, -c, a], axis=-1)
    X_adj = X_adj.reshape(shape[:-2] + (2, 2))
    X_inv = X_adj / det[..., np.newaxis, np.newaxis]

    return X_inv
