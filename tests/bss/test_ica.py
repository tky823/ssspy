from typing import Optional, Union, Callable, List, Dict, Any

import pytest
import numpy as np

from ssspy.bss.ica import GradICAbase, GradICA, GradLaplaceICA
from ssspy.bss.ica import NaturalGradICA, NaturalGradLaplaceICA
from ssspy.bss.ica import FastICA
from ssspy.utils.dataset import download_sample_speech_data
from dummy.callback import DummyCallback, dummy_function

max_samples = 8000
n_iter = 3

parameters_callbacks = [None, dummy_function, [DummyCallback(), dummy_function]]
parameters_is_holonomic = [True, False]
parameters_grad_ica = [
    (2, {}),
    (3, {"demix_filter": -np.eye(3)}),
]
parameters_fast_ica = [
    (2, {}),
    (3, {"demix_filter": -np.eye(3)}),
]


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_ica)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_grad_ica_base(
    n_sources: int,
    callbacks: Optional[Union[Callable[[GradICA], None], List[Callable[[GradICA], None]]]],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    def contrast_fn(x):
        return np.log(1 + np.exp(x))

    def score_fn(x):
        return 1 / (1 + np.exp(-x))

    ica = GradICAbase(contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks)

    print(ica)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_ica)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_grad_ica(
    n_sources: int,
    callbacks: Optional[Union[Callable[[GradICA], None], List[Callable[[GradICA], None]]]],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
        conv=False,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    def contrast_fn(x):
        return np.log(1 + np.exp(x))

    def score_fn(x):
        return 1 / (1 + np.exp(-x))

    ica = GradICA(
        contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks, is_holonomic=is_holonomic
    )
    waveform_est = ica(waveform_mix, n_iter=n_iter)

    assert waveform_mix.shape == waveform_est.shape

    print(ica)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_ica)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_natural_grad_ica(
    n_sources: int,
    callbacks: Optional[Union[Callable[[GradICA], None], List[Callable[[GradICA], None]]]],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
        conv=False,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    def contrast_fn(x):
        return np.log(1 + np.exp(x))

    def score_fn(x):
        return 1 / (1 + np.exp(-x))

    ica = NaturalGradICA(
        contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks, is_holonomic=is_holonomic
    )
    waveform_est = ica(waveform_mix, n_iter=n_iter)

    assert waveform_mix.shape == waveform_est.shape

    print(ica)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_ica)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_grad_laplace_ica(
    n_sources: int,
    callbacks: Optional[Union[Callable[[GradICA], None], List[Callable[[GradICA], None]]]],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
        conv=False,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    ica = GradLaplaceICA(callbacks=callbacks, is_holonomic=is_holonomic)
    waveform_est = ica(waveform_mix, n_iter=n_iter)

    assert waveform_mix.shape == waveform_est.shape

    print(ica)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_ica)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_natural_grad_laplace_ica(
    n_sources: int,
    callbacks: Optional[Union[Callable[[GradICA], None], List[Callable[[GradICA], None]]]],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
        conv=False,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    ica = NaturalGradLaplaceICA(callbacks=callbacks, is_holonomic=is_holonomic)
    waveform_est = ica(waveform_mix, n_iter=n_iter)

    assert waveform_mix.shape == waveform_est.shape

    print(ica)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_fast_ica)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
def test_fast_ica(
    n_sources: int,
    callbacks: Optional[Union[Callable[[FastICA], None], List[Callable[[FastICA], None]]]],
    reset_kwargs: Dict[Any, Any],
):
    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
        conv=False,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

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

    print(ica)
