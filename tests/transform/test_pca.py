import pytest
import numpy as np

from ssspy.transform import pca

parameters_pca = [(2, 257, 8), (3, 65, 12)]


@pytest.mark.parametrize("n_channels, n_bins, n_frames", parameters_pca)
def test_pca(n_channels: int, n_bins: int, n_frames: int):
    np.random.seed(111)

    real = np.random.randn(n_channels, n_bins, n_frames)
    imag = np.random.randn(n_channels, n_bins, n_frames)
    input = real + 1j * imag
    output = pca(input)

    assert input.shape == output.shape

    covariance = output[:, np.newaxis, :, :] * output[np.newaxis, :, :, :].conj()
    covariance = np.mean(covariance, axis=-1)
    covariance = covariance.transpose(2, 0, 1)
    mask = 1 - np.eye(n_channels)
    zero = np.zeros((n_bins, n_channels, n_channels))

    assert np.allclose(mask * covariance, zero)
