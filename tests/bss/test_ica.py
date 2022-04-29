from typing import Optional, Union, Callable, List

import pytest
import numpy as np

from ssspy.bss.ica import GradICA, NaturalGradICA, GradLaplaceICA, NaturalGradLaplaceICA
from tests.bss.create_dataset import create_sisec2011_mird_dataset

n_iter = 5


def dummy_function(_) -> None:
    pass


class DummyCallback:
    def __init__(self) -> None:
        pass

    def __call__(self, _) -> None:
        pass


parameters = [
    (2, None),
    (3, dummy_function),
    (2, [DummyCallback(), dummy_function]),
]


@pytest.mark.parametrize("n_sources, callbacks", parameters)
def test_grad_ica(
    n_sources: str,
    callbacks: Optional[Union[Callable[[GradICA], None], List[Callable[[GradICA], None]]]],
):
    np.random.seed(111)

    npz_path = create_sisec2011_mird_dataset(n_sources=n_sources)

    npz = np.load(npz_path)
    waveform_src = []

    for src_idx in range(n_sources):
        waveform_src.append(npz["src_{}".format(src_idx + 1)])

    waveform_src = np.stack(waveform_src, axis=1)  # (n_channels, n_sources, n_samples)
    waveform_mix = np.sum(waveform_src, axis=1)  # (n_channels, n_samples)

    ica = GradICA(callbacks=callbacks)
    ica.contrast_fn = lambda x: np.log(1 + np.exp(x))
    ica.score_fn = lambda x: 1 / (1 + np.exp(-x))

    waveform_est = ica(waveform_mix, n_iter=n_iter)

    assert waveform_mix.shape == waveform_est.shape


@pytest.mark.parametrize("n_sources, callbacks", parameters)
def test_natural_grad_ica(
    n_sources: str,
    callbacks: Optional[Union[Callable[[GradICA], None], List[Callable[[GradICA], None]]]],
):
    np.random.seed(111)

    npz_path = create_sisec2011_mird_dataset(n_sources=n_sources)

    npz = np.load(npz_path)
    waveform_src = []

    for src_idx in range(n_sources):
        waveform_src.append(npz["src_{}".format(src_idx + 1)])

    waveform_src = np.stack(waveform_src, axis=1)  # (n_channels, n_sources, n_samples)
    waveform_mix = np.sum(waveform_src, axis=1)  # (n_channels, n_samples)

    ica = NaturalGradICA(callbacks=callbacks)
    ica.contrast_fn = lambda x: np.log(1 + np.exp(x))
    ica.score_fn = lambda x: 1 / (1 + np.exp(-x))

    waveform_est = ica(waveform_mix, n_iter=n_iter)

    assert waveform_mix.shape == waveform_est.shape


@pytest.mark.parametrize("n_sources, callbacks", parameters)
def test_grad_laplace_ica(
    n_sources: str,
    callbacks: Optional[Union[Callable[[GradICA], None], List[Callable[[GradICA], None]]]],
):
    np.random.seed(111)

    npz_path = create_sisec2011_mird_dataset(n_sources=n_sources)

    npz = np.load(npz_path)
    waveform_src = []

    for src_idx in range(n_sources):
        waveform_src.append(npz["src_{}".format(src_idx + 1)])

    waveform_src = np.stack(waveform_src, axis=1)  # (n_channels, n_sources, n_samples)
    waveform_mix = np.sum(waveform_src, axis=1)  # (n_channels, n_samples)

    ica = GradLaplaceICA(callbacks=callbacks)
    waveform_est = ica(waveform_mix, n_iter=n_iter)

    assert waveform_mix.shape == waveform_est.shape


@pytest.mark.parametrize("n_sources, callbacks", parameters)
def test_natural_grad_laplace_ica(
    n_sources: str,
    callbacks: Optional[Union[Callable[[GradICA], None], List[Callable[[GradICA], None]]]],
):
    np.random.seed(111)

    npz_path = create_sisec2011_mird_dataset(n_sources=n_sources)

    npz = np.load(npz_path)
    waveform_src = []

    for src_idx in range(n_sources):
        waveform_src.append(npz["src_{}".format(src_idx + 1)])

    waveform_src = np.stack(waveform_src, axis=1)  # (n_channels, n_sources, n_samples)
    waveform_mix = np.sum(waveform_src, axis=1)  # (n_channels, n_samples)

    ica = NaturalGradLaplaceICA(callbacks=callbacks)
    waveform_est = ica(waveform_mix, n_iter=n_iter)

    assert waveform_mix.shape == waveform_est.shape
