import numpy as np


def inv2(X: np.ndarray) -> np.ndarray:
    r"""Compute inversion of 2x2 matrix.

    Args:
        X (numpy.ndarray):
            Complex matrix with shape of (*, 2, 2).

    Returns:
        numpy.ndarray:
            Inverse matrix of X whose shape is (*, 2, 2).
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
    X_inv = X_adj / det[..., None, None]

    return X_inv
