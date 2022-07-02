import numpy as np


def pca(input: np.ndarray):
    r"""
    Args:
        input (numpy.ndarray):
            Input tensor with shape of (n_channels, n_bins, n_frames).

    Returns:
        numpy.ndarray:
            Output tensor with shape of (n_channels, n_bins, n_frames).
    """
    if input.ndim == 3:
        X = input.transpose(1, 2, 0)
        covariance = np.mean(X[:, :, :, np.newaxis] * X[:, :, np.newaxis, :].conj(), axis=1)
        _, V = np.linalg.eigh(covariance)
        Y = X @ V.conj()
        output = Y.transpose(2, 0, 1)
    else:
        raise ValueError("The dimension of input is expected 3, but given {}.".format(input.ndim))

    return output
