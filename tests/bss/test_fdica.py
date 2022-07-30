from typing import Optional, Union, Callable, List, Dict, Any

import pytest
import numpy as np
import scipy.signal as ss

from ssspy.bss.fdica import GradFDICAbase, GradFDICA, GradLaplaceFDICA
from ssspy.bss.fdica import NaturalGradFDICA, NaturalGradLaplaceFDICA
from ssspy.bss.fdica import AuxFDICA, AuxLaplaceFDICA
from ssspy.utils.dataset import download_sample_speech_data
from tests.dummy.callback import DummyCallback, dummy_function

max_samples = 8000
n_fft = 512
hop_length = 256
n_bins = n_fft // 2 + 1
n_iter = 3

parameters_callbacks = [None, dummy_function, [DummyCallback(), dummy_function]]
parameters_is_holonomic = [True, False]
parameters_algorithm_spatial = ["IP", "IP1", "IP2"]
parameters_grad_fdica = [
    (2, {}),
    (3, {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},),
]
parameters_aux_fdica = [
    (2, {}),
    (3, {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},),
]


@pytest.mark.parametrize("callbacks", parameters_callbacks)
def test_grad_fdica_base(
    callbacks: Optional[Union[Callable[[GradFDICA], None], List[Callable[[GradFDICA], None]]]],
):
    np.random.seed(111)

    def contrast_fn(y):
        return 2 * np.abs(y)

    def score_fn(y):
        denominator = np.maximum(np.abs(y), 1e-10)
        return y / denominator

    fdica = GradFDICAbase(contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks)

    print(fdica)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_fdica)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_grad_fdica(
    n_sources: int,
    callbacks: Optional[Union[Callable[[GradFDICA], None], List[Callable[[GradFDICA], None]]]],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    def contrast_fn(y):
        return 2 * np.abs(y)

    def score_fn(y):
        denominator = np.maximum(np.abs(y), 1e-10)
        return y / denominator

    fdica = GradFDICA(
        contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks, is_holonomic=is_holonomic
    )
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(fdica)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_fdica)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_natural_grad_fdica(
    n_sources: int,
    callbacks: Optional[
        Union[Callable[[NaturalGradFDICA], None], List[Callable[[NaturalGradFDICA], None]]]
    ],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    def contrast_fn(y):
        return 2 * np.abs(y)

    def score_fn(y):
        denominator = np.maximum(np.abs(y), 1e-10)
        return y / denominator

    fdica = NaturalGradFDICA(
        contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks, is_holonomic=is_holonomic
    )
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(fdica)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_aux_fdica)
@pytest.mark.parametrize("algorithm_spatial", parameters_algorithm_spatial)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
def test_aux_fdica(
    n_sources: int,
    algorithm_spatial: str,
    callbacks: Optional[Union[Callable[[AuxFDICA], None], List[Callable[[AuxFDICA], None]]]],
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    def contrast_fn(y):
        return 2 * np.abs(y)

    def d_contrast_fn(y):
        return 2 * np.ones_like(y)

    fdica = AuxFDICA(
        algorithm_spatial=algorithm_spatial,
        contrast_fn=contrast_fn,
        d_contrast_fn=d_contrast_fn,
        callbacks=callbacks,
    )
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(fdica)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_fdica)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_grad_laplace_fdica(
    n_sources: int,
    callbacks: Optional[
        Union[Callable[[GradLaplaceFDICA], None], List[Callable[[GradLaplaceFDICA], None]]]
    ],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    fdica = GradLaplaceFDICA(callbacks=callbacks, is_holonomic=is_holonomic)
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(fdica)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_fdica)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_natural_grad_laplace_fdica(
    n_sources: int,
    callbacks: Optional[
        Union[
            Callable[[NaturalGradLaplaceFDICA], None],
            List[Callable[[NaturalGradLaplaceFDICA], None]],
        ]
    ],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    fdica = NaturalGradLaplaceFDICA(callbacks=callbacks, is_holonomic=is_holonomic)
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(fdica)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_aux_fdica)
@pytest.mark.parametrize("algorithm_spatial", parameters_algorithm_spatial)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
def test_aux_laplace_fdica(
    n_sources: int,
    algorithm_spatial: str,
    callbacks: Optional[
        Union[Callable[[AuxLaplaceFDICA], None], List[Callable[[AuxLaplaceFDICA], None]]]
    ],
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    fdica = AuxLaplaceFDICA(algorithm_spatial=algorithm_spatial, callbacks=callbacks)
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(fdica)
