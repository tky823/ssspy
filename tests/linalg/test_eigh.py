import pytest
import numpy as np

from ssspy.linalg import eigh

parameters_eigh = [(2, 4, 32), (5, 3, 16)]


@pytest.mark.parametrize("n_sources, n_channels, n_frames", parameters_eigh)
def test_eigh(n_sources: int, n_channels: int, n_frames: int):
    np.random.seed(111)

    shape = (n_sources, n_channels, n_frames)

    a = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    b = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    A = np.mean(a[:, :, np.newaxis, :] * a[:, np.newaxis, :, :].conj(), axis=-1)
    B = np.mean(b[:, :, np.newaxis, :] * b[:, np.newaxis, :, :].conj(), axis=-1)

    lamb, z = eigh(A, B)

    assert lamb.shape == (n_sources, n_channels)
    assert z.shape == (n_sources, n_channels, n_channels)
