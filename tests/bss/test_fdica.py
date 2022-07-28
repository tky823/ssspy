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
n_iter = 5

parameters_grad_fdica = [
    (2, None, True, {}),
    (
        3,
        dummy_function,
        False,
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
    (2, [DummyCallback(), dummy_function], False, {"demix_filter": None}),
]
parameters_aux_fdica = [
    (2, "IP", None, {}),
    (
        3,
        "IP1",
        dummy_function,
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
    (2, "IP2", [DummyCallback(), dummy_function], {"demix_filter": None}),
]


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic, reset_kwargs", parameters_grad_fdica)
def test_grad_fdica_base(
    n_sources: int,
    callbacks: Optional[Union[Callable[[GradFDICA], None], List[Callable[[GradFDICA], None]]]],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    def contrast_fn(y):
        return 2 * np.abs(y)

    def score_fn(y):
        denominator = np.maximum(np.abs(y), 1e-10)
        return y / denominator

    fdica = GradFDICAbase(
        contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks, is_holonomic=is_holonomic
    )

    print(fdica)


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic, reset_kwargs", parameters_grad_fdica)
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


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic, reset_kwargs", parameters_grad_fdica)
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


@pytest.mark.parametrize(
    "n_sources, algorithm_spatial, callbacks, reset_kwargs", parameters_aux_fdica
)
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


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic, reset_kwargs", parameters_grad_fdica)
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


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic, reset_kwargs", parameters_grad_fdica)
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


@pytest.mark.parametrize(
    "n_sources, algorithm_spatial, callbacks, reset_kwargs", parameters_aux_fdica
)
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
