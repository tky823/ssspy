import numpy as np


def whiten(input: np.ndarray):
    r"""
    Args:
        input (numpy.ndarray):
            Tensor with shape of (n_channels, n_samples)
            or (n_channels, n_bins, n_frames).

    Returns:
        numpy.ndarray:
            Whitened tensor.
            Tensor with shape of (n_channels, n_samples)
            or (n_channels, n_bins, n_frames).
    """
    n_channels = input.shape[0]

    if input.ndim == 2:
        X = input.transpose(1, 0)

        if np.iscomplexobj(input):
            covariance = np.mean(X[:, :, np.newaxis] * X[:, np.newaxis, :].conj(), axis=0)
            W, V = np.linalg.eigh(covariance)
            D_diag = 1 / np.sqrt(W)
            D_diag = np.diag(D_diag)
            V_Hermite = V.transpose(1, 0).conj()
            output = D_diag @ V_Hermite @ X.transpose(1, 0)
        else:
            covariance = np.mean(X[:, :, np.newaxis] * X[:, np.newaxis, :], axis=0)
            W, V = np.linalg.eigh(covariance)
            D_diag = 1 / np.sqrt(W)
            D_diag = np.diag(D_diag)
            V_transpose = V.transpose(1, 0)
            output = D_diag @ V_transpose @ X.transpose(1, 0)
    elif input.ndim == 3:
        X = input.transpose(1, 2, 0)

        if np.iscomplexobj(input):
            covariance = np.mean(X[:, :, :, np.newaxis] * X[:, :, np.newaxis, :].conj(), axis=1)
            W, V = np.linalg.eigh(covariance)
            D_diag = 1 / np.sqrt(W)
            D_diag = D_diag[:, :, np.newaxis]
            D_diag = D_diag * np.eye(n_channels)
            V_Hermite = V.transpose(0, 2, 1).conj()
            Y = D_diag @ V_Hermite @ X.transpose(0, 2, 1)
        else:
            covariance = np.mean(X[:, :, :, np.newaxis] * X[:, :, np.newaxis, :], axis=1)
            W, V = np.linalg.eigh(covariance)
            D_diag = 1 / np.sqrt(W)
            D_diag = D_diag[:, :, np.newaxis]
            D_diag = D_diag * np.eye(n_channels)
            V_transpose = V.transpose(0, 2, 1)
            Y = D_diag @ V_transpose @ X.transpose(0, 2, 1)

        output = Y.transpose(1, 0, 2)
    else:
        raise ValueError("The dimension of input is expected 3, but given {}.".format(input.ndim))

    return output
