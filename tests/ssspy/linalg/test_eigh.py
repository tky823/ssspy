import numpy as np
import pytest

from ssspy.linalg import eigh, eigh2

parameters_sources = [2, 5]
parameters_channels = [4, 3]
parameters_frames = [32, 16]
parameters_is_complex = [True, False]


@pytest.mark.parametrize("n_sources", parameters_sources)
@pytest.mark.parametrize("n_channels", parameters_channels)
@pytest.mark.parametrize("n_frames", parameters_frames)
@pytest.mark.parametrize("is_complex", parameters_is_complex)
def test_eigh(n_sources: int, n_channels: int, n_frames: int, is_complex: bool):
    np.random.seed(111)

    shape = (n_sources, n_channels, n_frames)

    if is_complex:
        a = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        A = np.mean(a[:, :, np.newaxis, :] * a[:, np.newaxis, :, :].conj(), axis=-1)
    else:
        a = np.random.randn(*shape)
        A = np.mean(a[:, :, np.newaxis, :] * a[:, np.newaxis, :, :], axis=-1)

    lamb, z = eigh(A)

    assert lamb.shape == (n_sources, n_channels)
    assert z.shape == (n_sources, n_channels, n_channels)

    assert np.allclose(A @ z, lamb[:, np.newaxis, :] * z)


@pytest.mark.parametrize("n_sources", parameters_sources)
@pytest.mark.parametrize("n_channels", parameters_channels)
@pytest.mark.parametrize("n_frames", parameters_frames)
@pytest.mark.parametrize("is_complex", parameters_is_complex)
def test_generalized_eigh(n_sources: int, n_channels: int, n_frames: int, is_complex: bool):
    np.random.seed(111)

    shape = (n_sources, n_channels, n_frames)

    if is_complex:
        a = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        b = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        A = np.mean(a[:, :, np.newaxis, :] * a[:, np.newaxis, :, :].conj(), axis=-1)
        B = np.mean(b[:, :, np.newaxis, :] * b[:, np.newaxis, :, :].conj(), axis=-1)
    else:
        a = np.random.randn(*shape)
        b = np.random.randn(*shape)
        A = np.mean(a[:, :, np.newaxis, :] * a[:, np.newaxis, :, :], axis=-1)
        B = np.mean(b[:, :, np.newaxis, :] * b[:, np.newaxis, :, :], axis=-1)

    lamb, z = eigh(A, B)

    assert lamb.shape == (n_sources, n_channels)
    assert z.shape == (n_sources, n_channels, n_channels)

    assert np.allclose(A @ z, lamb[:, np.newaxis, :] * (B @ z))


@pytest.mark.parametrize("n_sources", parameters_sources)
@pytest.mark.parametrize("n_frames", parameters_frames)
@pytest.mark.parametrize("is_complex", parameters_is_complex)
def test_eigh2(n_sources: int, n_frames: int, is_complex: bool):
    np.random.seed(111)

    shape = (n_sources, 2, n_frames)

    if is_complex:
        a = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        A = np.mean(a[:, :, np.newaxis, :] * a[:, np.newaxis, :, :].conj(), axis=-1)
    else:
        a = np.random.randn(*shape)
        A = np.mean(a[:, :, np.newaxis, :] * a[:, np.newaxis, :, :], axis=-1)

    lamb, z = eigh2(A)

    assert lamb.shape == (n_sources, 2)
    assert z.shape == (n_sources, 2, 2)

    assert np.allclose(A @ z, lamb[:, np.newaxis, :] * z)


@pytest.mark.parametrize("n_sources", parameters_sources)
@pytest.mark.parametrize("n_frames", parameters_frames)
@pytest.mark.parametrize("is_complex", parameters_is_complex)
def test_generalized_eigh2(n_sources: int, n_frames: int, is_complex: bool):
    np.random.seed(111)

    shape = (n_sources, 2, n_frames)

    if is_complex:
        a = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        b = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        A = np.mean(a[:, :, np.newaxis, :] * a[:, np.newaxis, :, :].conj(), axis=-1)
        B = np.mean(b[:, :, np.newaxis, :] * b[:, np.newaxis, :, :].conj(), axis=-1)
    else:
        a = np.random.randn(*shape)
        b = np.random.randn(*shape)
        A = np.mean(a[:, :, np.newaxis, :] * a[:, np.newaxis, :, :], axis=-1)
        B = np.mean(b[:, :, np.newaxis, :] * b[:, np.newaxis, :, :], axis=-1)

    lamb, z = eigh2(A, B)

    assert lamb.shape == (n_sources, 2)
    assert z.shape == (n_sources, 2, 2)

    assert np.allclose(A @ z, lamb[:, np.newaxis, :] * (B @ z))
