import numpy as np


def pca(input: np.ndarray) -> np.ndarray:
    r"""
    Args:
        input (numpy.ndarray):
            If input is 3D complex tensor, it is regarded as \
            (n_channels, n_bins, n_frames).
            If input is 3D real tensor, it is regarded as \
            (batch_size, n_channels, n_samples).
            If input is 4D complex tensor, it is regarded as \
            (batch_size, n_channels, n_bins, n_frames).

    Returns:
        numpy.ndarray:
            Output tensor. The type (real or complex) and shape is same as input.
    """
    if input.ndim == 3:
        if np.iscomplexobj(input):
            X = input.transpose(1, 2, 0)
            covariance = np.mean(X[:, :, :, np.newaxis] * X[:, :, np.newaxis, :].conj(), axis=1)
            _, V = np.linalg.eigh(covariance)
            Y = X @ V.conj()
            output = Y.transpose(2, 0, 1)
        else:
            X = input.transpose(0, 2, 1)
            covariance = np.mean(X[:, :, :, np.newaxis] * X[:, :, np.newaxis, :], axis=1)
            _, V = np.linalg.eigh(covariance)
            Y = X @ V
            output = Y.transpose(0, 2, 1)
    elif input.ndim == 4:
        if np.iscomplexobj(input):
            X = input.transpose(0, 2, 3, 1)
            covariance = np.mean(
                X[:, :, :, :, np.newaxis] * X[:, :, :, np.newaxis, :].conj(), axis=2
            )
            _, V = np.linalg.eigh(covariance)
            Y = X @ V.conj()
            output = Y.transpose(0, 3, 1, 2)
        else:
            raise ValueError("Complex tensor is expected, but given real tensor.")
    else:
        raise ValueError("The dimension of input is expected 3, but given {}.".format(input.ndim))

    return output
