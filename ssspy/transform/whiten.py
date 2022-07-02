import numpy as np


def whiten(input: np.ndarray):
    r"""
    Args:
        input (numpy.ndarray):
            Input tensor with shape of (n_channels, n_bins, n_frames).

    Returns:
        numpy.ndarray:
            Output tensor with shape of (n_channels, n_bins, n_frames).
    """
    if input.ndim == 3:
        n_channels, _, _ = input.shape
        X = input.transpose(1, 2, 0)
        covariance = np.mean(X[:, :, :, np.newaxis] * X[:, :, np.newaxis, :].conj(), axis=1)
        W, V = np.linalg.eigh(covariance)
        D_diag = 1 / np.sqrt(W)
        D_diag = D_diag[:, :, np.newaxis]
        D_diag = D_diag * np.eye(n_channels)
        V_Hermite = V.transpose(0, 2, 1).conj()
        Y = D_diag @ V_Hermite @ X.transpose(0, 2, 1)
        output = Y.transpose(1, 0, 2)
    else:
        raise ValueError("The dimension of input is expected 3, but given {}.".format(input.ndim))

    return output
