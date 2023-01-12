import numpy as np


def softmax(X: np.ndarray, axis: int = None) -> np.ndarray:
    r"""Compute softmax values.

    Args:
        X (np.ndarray):
            Elements to compute softmax.
        axis (int or tuple[int], optional):
            Axis or axes over which the sum is performed.
            Default: ``None``.

    Returns:
        np.ndarray of softmax values.
    """
    vmax = np.max(X, axis=axis, keepdims=True)
    Y = X - vmax
    exp = np.exp(Y)
    v = exp / np.sum(exp, axis=axis, keepdims=True)

    return v
