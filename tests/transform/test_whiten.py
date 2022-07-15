import pytest
import numpy as np

from ssspy.transform import whiten

parameters_batch_size = [1, 4]
parameters_n_channels = [2, 3]
parameters_whiten_real = [10, 20]
parameters_whiten_complex = [(2049, 8), (513, 12)]


@pytest.mark.parametrize("n_channels", parameters_n_channels)
@pytest.mark.parametrize("n_samples", parameters_whiten_real)
def test_whiten_real_2d(n_channels: int, n_samples: int):
    np.random.seed(111)

    input = np.random.randn(n_channels, n_samples)
    output = whiten(input)

    assert input.shape == output.shape

    covariance = output[:, np.newaxis, :] * output[np.newaxis, :, :]
    covariance = np.mean(covariance, axis=-1)
    eye = np.eye(n_channels)

    assert np.allclose(covariance, eye)


@pytest.mark.parametrize("batch_size", parameters_batch_size)
@pytest.mark.parametrize("n_channels", parameters_n_channels)
@pytest.mark.parametrize("n_samples", parameters_whiten_real)
def test_whiten_real_3d(batch_size: int, n_channels: int, n_samples: int):
    np.random.seed(111)

    input = np.random.randn(batch_size, n_channels, n_samples)
    output = whiten(input)

    assert input.shape == output.shape

    covariance = output[:, :, np.newaxis, :] * output[:, np.newaxis, :, :]
    covariance = np.mean(covariance, axis=-1)
    eye = np.eye(n_channels)

    assert np.allclose(covariance, eye)


@pytest.mark.parametrize("n_channels", parameters_n_channels)
@pytest.mark.parametrize("n_bins, n_frames", parameters_whiten_complex)
def test_whiten_complex_3d(n_channels: int, n_bins: int, n_frames: int):
    np.random.seed(111)

    real = np.random.randn(n_channels, n_bins, n_frames)
    imag = np.random.randn(n_channels, n_bins, n_frames)
    input = real + 1j * imag
    output = whiten(input)

    assert input.shape == output.shape

    covariance = output[:, np.newaxis, :, :] * output[np.newaxis, :, :, :].conj()
    covariance = np.mean(covariance, axis=-1)
    covariance = covariance.transpose(2, 0, 1)
    eye = np.eye(n_channels)
    eye = np.tile(eye, reps=(n_bins, 1, 1))

    assert np.allclose(covariance, eye)


@pytest.mark.parametrize("batch_size", parameters_batch_size)
@pytest.mark.parametrize("n_channels", parameters_n_channels)
@pytest.mark.parametrize("n_bins, n_frames", parameters_whiten_complex)
def test_whiten_complex_4d(batch_size: int, n_channels: int, n_bins: int, n_frames: int):
    np.random.seed(111)

    real = np.random.randn(batch_size, n_channels, n_bins, n_frames)
    imag = np.random.randn(batch_size, n_channels, n_bins, n_frames)
    input = real + 1j * imag
    output = whiten(input)

    assert input.shape == output.shape

    covariance = output[:, :, np.newaxis, :, :] * output[:, np.newaxis, :, :, :].conj()
    covariance = np.mean(covariance, axis=-1)
    covariance = covariance.transpose(0, 3, 1, 2)
    eye = np.eye(n_channels)
    eye = np.tile(eye, reps=(batch_size, n_bins, 1, 1))

    assert np.allclose(covariance, eye)
