import numpy as np


def _kummer(kappa: np.ndarray) -> np.ndarray:
    """Kummer function M(1, N; X).

    Args:
        kappa (np.ndarray): Concentration parameter of shape (n_sources,).

    Returns:
        np.ndarray of computed values of shape (n_sources,).

    """
    n_sources = kappa.shape[0]
    indices = np.arange(1, n_sources)
    cumprod = np.cumprod(indices, axis=0)
    kappa_l = kappa ** indices[:, np.newaxis]
    scale = cumprod[-1] / kappa_l[-1]
    exp_kappa = np.exp(kappa)
    terms = kappa_l / cumprod[:, np.newaxis]
    k = scale * (exp_kappa - np.sum(terms[:-1], axis=0) - 1)

    return k
