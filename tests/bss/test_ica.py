from typing import Optional, Union, Callable, List

import pytest
import numpy as np

from ssspy.bss.ica import GradICA, NaturalGradICA, GradLaplaceICA, NaturalGradLaplaceICA
from ssspy.bss.ica import FastICA
from tests.bss.create_dataset import create_sisec2011_mird_dataset

n_iter = 5


def dummy_function(_) -> None:
    pass


class DummyCallback:
    def __init__(self) -> None:
        pass

    def __call__(self, _) -> None:
        pass


parameters_grad_ica = [
    (2, None, True),
    (3, dummy_function, False),
    (2, [DummyCallback(), dummy_function], False),
]
parameters_fast_ica = [
    (2, None),
    (3, dummy_function),
    (2, [DummyCallback(), dummy_function]),
]


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic", parameters_grad_ica)
def test_grad_ica(
    n_sources: int,
    callbacks: Optional[Union[Callable[[GradICA], None], List[Callable[[GradICA], None]]]],
    is_holonomic: bool,
):
    np.random.seed(111)

    npz_path = create_sisec2011_mird_dataset(n_sources=n_sources)

    npz = np.load(npz_path)
    waveform_src = []

    for src_idx in range(n_sources):
        waveform_src.append(npz["src_{}".format(src_idx + 1)])

    waveform_src = np.stack(waveform_src, axis=1)  # (n_channels, n_sources, n_samples)
    waveform_mix = np.sum(waveform_src, axis=1)  # (n_channels, n_samples)

    def contrast_fn(x):
        return np.log(1 + np.exp(x))

    def score_fn(x):
        return 1 / (1 + np.exp(-x))

    ica = GradICA(
        contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks, is_holonomic=is_holonomic
    )

    waveform_est = ica(waveform_mix, n_iter=n_iter)

    assert waveform_mix.shape == waveform_est.shape


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic", parameters_grad_ica)
def test_natural_grad_ica(
    n_sources: int,
    callbacks: Optional[Union[Callable[[GradICA], None], List[Callable[[GradICA], None]]]],
    is_holonomic: bool,
):
    np.random.seed(111)

    npz_path = create_sisec2011_mird_dataset(n_sources=n_sources)

    npz = np.load(npz_path)
    waveform_src = []

    for src_idx in range(n_sources):
        waveform_src.append(npz["src_{}".format(src_idx + 1)])

    waveform_src = np.stack(waveform_src, axis=1)  # (n_channels, n_sources, n_samples)
    waveform_mix = np.sum(waveform_src, axis=1)  # (n_channels, n_samples)

    def contrast_fn(x):
        return np.log(1 + np.exp(x))

    def score_fn(x):
        return 1 / (1 + np.exp(-x))

    ica = NaturalGradICA(
        contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks, is_holonomic=is_holonomic
    )

    waveform_est = ica(waveform_mix, n_iter=n_iter)

    assert waveform_mix.shape == waveform_est.shape


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic", parameters_grad_ica)
def test_grad_laplace_ica(
    n_sources: int,
    callbacks: Optional[Union[Callable[[GradICA], None], List[Callable[[GradICA], None]]]],
    is_holonomic: bool,
):
    np.random.seed(111)

    npz_path = create_sisec2011_mird_dataset(n_sources=n_sources)

    npz = np.load(npz_path)
    waveform_src = []

    for src_idx in range(n_sources):
        waveform_src.append(npz["src_{}".format(src_idx + 1)])

    waveform_src = np.stack(waveform_src, axis=1)  # (n_channels, n_sources, n_samples)
    waveform_mix = np.sum(waveform_src, axis=1)  # (n_channels, n_samples)

    ica = GradLaplaceICA(callbacks=callbacks, is_holonomic=is_holonomic)
    waveform_est = ica(waveform_mix, n_iter=n_iter)

    assert waveform_mix.shape == waveform_est.shape


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic", parameters_grad_ica)
def test_natural_grad_laplace_ica(
    n_sources: int,
    callbacks: Optional[Union[Callable[[GradICA], None], List[Callable[[GradICA], None]]]],
    is_holonomic: bool,
):
    np.random.seed(111)

    npz_path = create_sisec2011_mird_dataset(n_sources=n_sources)

    npz = np.load(npz_path)
    waveform_src = []

    for src_idx in range(n_sources):
        waveform_src.append(npz["src_{}".format(src_idx + 1)])

    waveform_src = np.stack(waveform_src, axis=1)  # (n_channels, n_sources, n_samples)
    waveform_mix = np.sum(waveform_src, axis=1)  # (n_channels, n_samples)

    ica = NaturalGradLaplaceICA(callbacks=callbacks, is_holonomic=is_holonomic)
    waveform_est = ica(waveform_mix, n_iter=n_iter)

    assert waveform_mix.shape == waveform_est.shape


@pytest.mark.parametrize("n_sources, callbacks", parameters_fast_ica)
def test_fast_ica(
    n_sources: int,
    callbacks: Optional[Union[Callable[[FastICA], None], List[Callable[[FastICA], None]]]],
):
    np.random.seed(111)

    npz_path = create_sisec2011_mird_dataset(n_sources=n_sources)

    npz = np.load(npz_path)
    waveform_src = []

    for src_idx in range(n_sources):
        waveform_src.append(npz["src_{}".format(src_idx + 1)])

    waveform_src = np.stack(waveform_src, axis=1)  # (n_channels, n_sources, n_samples)
    waveform_mix = np.sum(waveform_src, axis=1)  # (n_channels, n_samples)

    def contrast_fn(x):
        return np.log(1 + np.exp(x))

    def score_fn(x):
        return 1 / (1 + np.exp(-x))

    def d_score_fn(x):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)

    ica = FastICA(
        contrast_fn=contrast_fn, score_fn=score_fn, d_score_fn=d_score_fn, callbacks=callbacks
    )

    waveform_est = ica(waveform_mix, n_iter=n_iter)

    assert waveform_mix.shape == waveform_est.shape
