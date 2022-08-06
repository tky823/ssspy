import numpy as np

__all__ = ["neg_log", "logdet"]


def neg_log(x: np.ndarray, step_size: float = 1):
    r"""Proxmal operator of negative logarithm function.

    Proximal operator of :math:`-\log(x)` is defined as follows:

    .. math::
        \mathrm{prox}_{-\mu\log}(x)
        = \frac{x + \sqrt{x + 4\mu}}{2}

    Args:
        x (np.ndarray):
            Shape is (n_bins, n_sources, n_channels).
        step_size (float):
            Step size parameter. Default: 1.

    Returns:
        np.ndarray:
            Proximal operator of negative logarithm function.
    """
    assert np.all(x >= 0)

    output = (x + np.sqrt(x + 4 * step_size)) / 2

    return output


def logdet(X: np.ndarray, step_size=1):
    r"""Proxmal operator of log-determinant.

    :math:`X\in\mathbb{C}^{N\times M}`

    .. math::
        \mathrm{prox}_{-\mu\log}(\boldsymbol{X})
        &= \boldsymbol{U}\tilde{\boldsymbol{\Sigma}}\boldsymbol{V}^{\mathsf{H}} \\
        \tilde{\boldsymbol{\Sigma}}
        &= \mathrm{diag}(\mathrm{prox}_{-\mu\log}(\sigma_{1}),
        \ldots,\mathrm{prox}_{-\mu\log}(\sigma_{M}))

    Args:
        X (np.ndarray):
            Shape is (n_bins, n_sources, n_channels).
        step_size (float):
            Step size parameter. Default: 1.

    Returns:
        np.ndarray:
            Proxmal operator of log-determinant.
    """
    n_channels = X.shape[-1]

    U, Sigma, V = np.linalg.svd(X)
    Sigma = neg_log(Sigma, step_size=step_size)
    Sigma = Sigma[..., np.newaxis] * np.eye(n_channels)
    USV = U @ Sigma @ V

    return USV