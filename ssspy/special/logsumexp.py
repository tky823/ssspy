import numpy as np


def logsumexp(X: np.ndarray, axis: int = None, keepdims: bool = False) -> np.ndarray:
    r"""Compute log-sum-exp values.

    Args:
        X (np.ndarray):
            Elements to compute log-sum-exp.
        axis (int or tuple[int], optional):
            Axis or axes over which the sum is performed.
            Default: ``None``.
        keepdims (bool):
            If ``True`` is given, ``axis`` dimension(s) is reduced.
            Default: ``False``.

    Returns:
        np.ndarray of log-sum-exp values.

    Examples:

        .. code-block:: python

            >>> import numpy as np

            >>> X = np.array([[1, 2, 3], [4, 5, 6]])
            >>> logsumexp(X, axis=0)
            array([4.04858735, 5.04858735, 6.04858735])
            >>> logsumexp(X, axis=1)
            array([3.40760596, 6.40760596])
    """
    vmax = np.max(X, axis=axis, keepdims=True)
    exp = np.exp(X - vmax)
    sum_exp = exp.sum(axis=axis, keepdims=True)
    v = np.log(sum_exp) + vmax

    if not keepdims:
        v = np.squeeze(v, axis=axis)

    return v
