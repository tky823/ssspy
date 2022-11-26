from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import pytest

from ssspy.bss._flooring import add_flooring, max_flooring
from ssspy.bss._select_pair import combination_pair_selector, sequential_pair_selector
from ssspy.bss._update_spatial_model import (
    update_by_ip1,
    update_by_ip2,
    update_by_ip2_one_pair,
    update_by_iss1,
    update_by_iss2,
)


def negative_pair_selector(n_sources):
    for m in range(n_sources):
        m, n = m % n_sources, (m + 1) % n_sources
        m, n = m - n_sources, n - n_sources

        yield m, n


parameters = [(31, 20)]
parameters_n_sources = [2, 3]
parameters_flooring_fn = [max_flooring, add_flooring, None]
parameters_pair_selector = [
    sequential_pair_selector,
    combination_pair_selector,
    negative_pair_selector,
    None,
]


@pytest.mark.parametrize("n_bins, n_frames", parameters)
@pytest.mark.parametrize("n_sources", parameters_n_sources)
@pytest.mark.parametrize("flooring_fn", parameters_flooring_fn)
def test_update_by_ip1(
    n_bins: int,
    n_frames: int,
    n_sources: int,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]],
):
    n_channels = n_sources

    rng = np.random.default_rng(42)

    varphi = 1 / rng.random((n_bins, n_frames))
    X = rng.standard_normal((n_channels, n_bins, n_frames))
    real = rng.standard_normal((n_bins, n_sources, n_sources))
    imag = rng.standard_normal((n_bins, n_sources, n_sources))
    W = real + 1j * imag

    XX_Hermite = X[:, np.newaxis, :, :] * X[np.newaxis, :, :, :].conj()
    XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)
    GXX = varphi[:, np.newaxis, np.newaxis, :] * XX_Hermite[:, np.newaxis, :, :, :]
    U = np.mean(GXX, axis=-1)
    W_updated = update_by_ip1(W, U, flooring_fn=flooring_fn)

    assert W_updated.shape == W.shape


@pytest.mark.parametrize("n_bins, n_frames", parameters)
@pytest.mark.parametrize("n_sources", parameters_n_sources)
@pytest.mark.parametrize("flooring_fn", parameters_flooring_fn)
@pytest.mark.parametrize("pair_selector", parameters_pair_selector)
def test_update_by_ip2(
    n_bins: int,
    n_frames: int,
    n_sources: int,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    pair_selector: Optional[Callable[[int], Iterable[Tuple[int, int]]]],
):
    n_channels = n_sources

    rng = np.random.default_rng(42)

    varphi = 1 / rng.random((n_bins, n_frames))
    X = rng.standard_normal((n_channels, n_bins, n_frames))
    real = rng.standard_normal((n_bins, n_sources, n_sources))
    imag = rng.standard_normal((n_bins, n_sources, n_sources))
    W = real + 1j * imag

    XX_Hermite = X[:, np.newaxis, :, :] * X[np.newaxis, :, :, :].conj()
    XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)
    GXX = varphi[:, np.newaxis, np.newaxis, :] * XX_Hermite[:, np.newaxis, :, :, :]
    U = np.mean(GXX, axis=-1)
    W_updated = update_by_ip2(W, U, flooring_fn=flooring_fn, pair_selector=pair_selector)

    assert W_updated.shape == W.shape


@pytest.mark.parametrize("n_bins, n_frames", parameters)
@pytest.mark.parametrize("n_sources", parameters_n_sources)
@pytest.mark.parametrize("flooring_fn", parameters_flooring_fn)
def test_update_by_ip2_one_pair(
    n_bins: int,
    n_frames: int,
    n_sources: int,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]],
):
    n_channels = n_sources

    rng = np.random.default_rng(42)

    varphi = 1 / rng.random((2, n_bins, n_frames))
    X = rng.standard_normal((n_channels, n_bins, n_frames))
    real = rng.standard_normal((n_bins, n_sources, n_channels))
    imag = rng.standard_normal((n_bins, n_sources, n_channels))
    W = real + 1j * imag
    XX = X[:, np.newaxis] * X[np.newaxis, :].conj()
    GXX = np.mean(varphi[:, np.newaxis, np.newaxis, :, :] * XX[np.newaxis, :, :, :, :], axis=-1)
    GXX = GXX.transpose(3, 0, 1, 2)

    W_updated = update_by_ip2_one_pair(W, GXX, pair=(1, 0), flooring_fn=flooring_fn)

    assert W_updated.shape == (n_bins, 2, n_channels)


@pytest.mark.parametrize("n_bins, n_frames", parameters)
@pytest.mark.parametrize("n_sources", parameters_n_sources)
@pytest.mark.parametrize("flooring_fn", parameters_flooring_fn)
def test_update_by_iss1(
    n_bins: int,
    n_frames: int,
    n_sources: int,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]],
):
    rng = np.random.default_rng(42)

    varphi = 1 / rng.random((n_sources, n_bins, n_frames))
    real = rng.standard_normal((n_sources, n_bins, n_frames))
    imag = rng.standard_normal((n_sources, n_bins, n_frames))
    Y = real + 1j * imag

    Y_updated = update_by_iss1(Y, varphi, flooring_fn=flooring_fn)

    assert Y_updated.shape == Y.shape


@pytest.mark.parametrize("n_bins, n_frames", parameters)
@pytest.mark.parametrize("n_sources", parameters_n_sources)
@pytest.mark.parametrize("flooring_fn", parameters_flooring_fn)
@pytest.mark.parametrize("pair_selector", parameters_pair_selector)
def test_update_by_iss2(
    n_bins: int,
    n_frames: int,
    n_sources: int,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    pair_selector: Optional[Callable[[int], Iterable[Tuple[int, int]]]],
):
    rng = np.random.default_rng(42)

    varphi = 1 / rng.random((n_sources, n_bins, n_frames))
    real = rng.standard_normal((n_sources, n_bins, n_frames))
    imag = rng.standard_normal((n_sources, n_bins, n_frames))
    Y = real + 1j * imag

    Y_updated = update_by_iss2(Y, varphi, flooring_fn=flooring_fn, pair_selector=pair_selector)

    assert Y_updated.shape == Y.shape
