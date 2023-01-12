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
    """
    vmax = np.max(X, axis=axis, keepdims=True)
    exp = np.exp(X - vmax)
    sum_exp = exp.sum(axis=axis, keepdims=True)
    v = np.log(sum_exp) + vmax

    if not keepdims:
        v = np.squeeze(v, axis=axis)

    return v
