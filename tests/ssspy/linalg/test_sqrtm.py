import numpy as np
import pytest

from ssspy.linalg import invsqrtmh, sqrtmh

parameters_sources = [2]
parameters_channels = [3, 4]
parameters_frames = [32]
parameters_is_complex = [True, False]
parameters_is_flooring = [True, False]


@pytest.mark.parametrize("n_sources", parameters_sources)
@pytest.mark.parametrize("n_channels", parameters_channels)
@pytest.mark.parametrize("n_frames", parameters_frames)
@pytest.mark.parametrize("is_complex", parameters_is_complex)
def test_sqrtmh(n_sources: int, n_channels: int, n_frames: int, is_complex: bool):
    rng = np.random.default_rng(0)

    shape = (n_sources, n_channels, n_frames)

    if is_complex:
        x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        X = np.mean(x[:, :, np.newaxis, :] * x[:, np.newaxis, :, :].conj(), axis=-1)
    else:
        x = rng.standard_normal(shape)
        X = np.mean(x[:, :, np.newaxis, :] * x[:, np.newaxis, :, :], axis=-1)

    X_sqrt = sqrtmh(X)

    assert np.allclose(X, X_sqrt @ X_sqrt)


@pytest.mark.parametrize("n_sources", parameters_sources)
@pytest.mark.parametrize("n_channels", parameters_channels)
@pytest.mark.parametrize("n_frames", parameters_frames)
@pytest.mark.parametrize("is_complex", parameters_is_complex)
@pytest.mark.parametrize("is_flooring", parameters_is_flooring)
def test_invsqrtmh(
    n_sources: int, n_channels: int, n_frames: int, is_complex: bool, is_flooring: bool
):
    rng = np.random.default_rng(0)

    shape = (n_sources, n_channels, n_frames)

    if is_complex:
        x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        X = np.mean(x[:, :, np.newaxis, :] * x[:, np.newaxis, :, :].conj(), axis=-1)
    else:
        x = rng.standard_normal(shape)
        X = np.mean(x[:, :, np.newaxis, :] * x[:, np.newaxis, :, :], axis=-1)

    if is_flooring:
        X_invsqrt = invsqrtmh(X, flooring_fn=lambda x: np.maximum(x, 1e-12))
    else:
        X_invsqrt = invsqrtmh(X)

    X_sqrt = np.linalg.inv(X_invsqrt)

    assert np.allclose(X, X_sqrt @ X_sqrt)
