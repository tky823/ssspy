import numpy as np


def pca(input: np.ndarray, ascend: bool = True) -> np.ndarray:
    r"""Apply principal component analysis (PCA).

    Args:
        input (numpy.ndarray):
            Input tensor.
        ascend (bool):
            If ``ascend=True``, first channel corresponds to first principle component. \
            Otherwise, last channel corresponds to first principle component.

    Returns:
        numpy.ndarray:
            Output tensor. The type (real or complex) and shape are same as input.

    .. note::
        - If ``input`` is 2D real tensor, it is regarded as (n_channels, n_samples).
        - If ``input`` is 3D complex tensor, it is regarded as (n_channels, n_bins, n_frames).
        - If ``input`` is 3D real tensor, it is regarded as (batch_size, n_channels, n_samples).
        - If ``input`` is 4D complex tensor, it is regarded as \
          (batch_size, n_channels, n_bins, n_frames).

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from ssspy.transform import pca

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> n_sources = n_channels
            >>> rng = np.random.default_rng(42)

            >>> spectrogram_mix = \
            ...     rng.standard_normal((n_channels, n_bins, n_frames)) \
            ...     + 1j * rng.standard_normal((n_channels, n_bins, n_frames))
            >>> spectrogram_mix_ortho = pca(spectrogram_mix)
            >>> spectrogram_mix_ortho.shape
            (2, 2049, 128)
    """
    if input.ndim == 2:
        if np.iscomplexobj(input):
            raise ValueError("Real tensor is expected, but given complex tensor.")
        else:
            X = input.transpose(1, 0)
            covariance = np.mean(X[:, :, np.newaxis] * X[:, np.newaxis, :], axis=0)
            _, V = np.linalg.eigh(covariance)

            if ascend:
                V = V[..., ::-1]

            Y = X @ V
            output = Y.transpose(1, 0)
    elif input.ndim == 3:
        if np.iscomplexobj(input):
            X = input.transpose(1, 2, 0)
            covariance = np.mean(X[:, :, :, np.newaxis] * X[:, :, np.newaxis, :].conj(), axis=1)
            _, V = np.linalg.eigh(covariance)

            if ascend:
                V = V[..., ::-1]

            Y = X @ V.conj()
            output = Y.transpose(2, 0, 1)
        else:
            X = input.transpose(0, 2, 1)
            covariance = np.mean(X[:, :, :, np.newaxis] * X[:, :, np.newaxis, :], axis=1)
            _, V = np.linalg.eigh(covariance)

            if ascend:
                V = V[..., ::-1]

            Y = X @ V
            output = Y.transpose(0, 2, 1)
    elif input.ndim == 4:
        if np.iscomplexobj(input):
            X = input.transpose(0, 2, 3, 1)
            covariance = np.mean(
                X[:, :, :, :, np.newaxis] * X[:, :, :, np.newaxis, :].conj(), axis=2
            )
            _, V = np.linalg.eigh(covariance)

            if ascend:
                V = V[..., ::-1]

            Y = X @ V.conj()
            output = Y.transpose(0, 3, 1, 2)
        else:
            raise ValueError("Complex tensor is expected, but given real tensor.")
    else:
        raise ValueError(
            "The dimension of input is expected 3 or 4, but given {}.".format(input.ndim)
        )

    return output
