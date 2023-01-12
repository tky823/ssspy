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

    Examples:

        .. code-block:: python

            >>> import numpy as np

            >>> X = np.array([[1, 2, 3], [4, 5, 6]])
            >>> softmax(X, axis=0)
            array([[0.04742587, 0.04742587, 0.04742587],
                [0.95257413, 0.95257413, 0.95257413]])
            >>> softmax(X, axis=1)
            array([[0.09003057, 0.24472847, 0.66524096],
                [0.09003057, 0.24472847, 0.66524096]])
    """
    vmax = np.max(X, axis=axis, keepdims=True)
    Y = X - vmax
    exp = np.exp(Y)
    v = exp / np.sum(exp, axis=axis, keepdims=True)

    return v
