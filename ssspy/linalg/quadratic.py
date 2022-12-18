import numpy as np


def quadratic(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    r"""Compute values of quadratic forms.

    Args:
        X (np.ndarray):
            Input vectors with shape of (\*, n_channels).
        A (np.ndarray):
            Input matrices with shape of (\*, n_channels, n_channels).

    Returns:
        Computed values of quadratic forms.
        The shape is (\*,).
    """
    if np.iscomplexobj(X):
        X_Hermite = X.conj()
    else:
        X_Hermite = X

    Y = X_Hermite[..., np.newaxis, :] @ A @ X[..., np.newaxis]
    Y = Y[..., 0, 0]

    return Y
