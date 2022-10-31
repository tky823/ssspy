import numpy as np
import pytest

from ssspy.transform import pca

parameters_ascend = [True, False]
parameters_batch_size = [1, 4]
parameters_n_channels = [2, 3]
parameters_pca_real = [10, 20]
parameters_pca_complex = [(257, 8), (65, 12)]


@pytest.mark.parametrize("ascend", parameters_ascend)
@pytest.mark.parametrize("n_channels", parameters_n_channels)
@pytest.mark.parametrize("n_samples", parameters_pca_real)
def test_pca_real_2d(ascend: bool, n_channels: int, n_samples: int):
    np.random.seed(111)

    input = np.random.randn(n_channels, n_samples)
    output = pca(input, ascend=ascend)

    assert input.shape == output.shape

    covariance = output[:, np.newaxis, :] * output[np.newaxis, :, :]
    covariance = np.mean(covariance, axis=-1)
    mask = 1 - np.eye(n_channels)
    zero = np.zeros((n_channels, n_channels))

    assert np.allclose(mask * covariance, zero)


@pytest.mark.parametrize("ascend", parameters_ascend)
@pytest.mark.parametrize("batch_size", parameters_batch_size)
@pytest.mark.parametrize("n_channels", parameters_n_channels)
@pytest.mark.parametrize("n_samples", parameters_pca_real)
def test_pca_real_3d(ascend: bool, batch_size: int, n_channels: int, n_samples: int):
    np.random.seed(111)

    input = np.random.randn(batch_size, n_channels, n_samples)
    output = pca(input, ascend=ascend)

    assert input.shape == output.shape

    covariance = output[:, :, np.newaxis, :] * output[:, np.newaxis, :, :]
    covariance = np.mean(covariance, axis=-1)
    mask = 1 - np.eye(n_channels)
    zero = np.zeros((batch_size, n_channels, n_channels))

    assert np.allclose(mask * covariance, zero)


@pytest.mark.parametrize("ascend", parameters_ascend)
@pytest.mark.parametrize("n_channels", parameters_n_channels)
@pytest.mark.parametrize("n_bins, n_frames", parameters_pca_complex)
def test_pca_complex_3d(ascend: bool, n_channels: int, n_bins: int, n_frames: int):
    np.random.seed(111)

    real = np.random.randn(n_channels, n_bins, n_frames)
    imag = np.random.randn(n_channels, n_bins, n_frames)
    input = real + 1j * imag
    output = pca(input, ascend=ascend)

    assert input.shape == output.shape

    covariance = output[:, np.newaxis, :, :] * output[np.newaxis, :, :, :].conj()
    covariance = np.mean(covariance, axis=-1)
    covariance = covariance.transpose(2, 0, 1)
    mask = 1 - np.eye(n_channels)
    zero = np.zeros((n_bins, n_channels, n_channels))

    assert np.allclose(mask * covariance, zero)


@pytest.mark.parametrize("ascend", parameters_ascend)
@pytest.mark.parametrize("batch_size", parameters_batch_size)
@pytest.mark.parametrize("n_channels", parameters_n_channels)
@pytest.mark.parametrize("n_bins, n_frames", parameters_pca_complex)
def test_pca_complex_4d(ascend: bool, batch_size: int, n_channels: int, n_bins: int, n_frames: int):
    np.random.seed(111)

    real = np.random.randn(batch_size, n_channels, n_bins, n_frames)
    imag = np.random.randn(batch_size, n_channels, n_bins, n_frames)
    input = real + 1j * imag
    output = pca(input, ascend=ascend)

    assert input.shape == output.shape

    covariance = output[:, :, np.newaxis, :, :] * output[:, np.newaxis, :, :, :].conj()
    covariance = np.mean(covariance, axis=-1)
    covariance = covariance.transpose(0, 3, 1, 2)
    mask = 1 - np.eye(n_channels)
    zero = np.zeros((batch_size, n_bins, n_channels, n_channels))

    assert np.allclose(mask * covariance, zero)
