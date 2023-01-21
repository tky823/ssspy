import os
import sys
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pytest
import scipy.signal as ss

from ssspy.bss.fdica import (
    AuxFDICA,
    AuxLaplaceFDICA,
    GradFDICA,
    GradFDICABase,
    GradLaplaceFDICA,
    NaturalGradFDICA,
    NaturalGradLaplaceFDICA,
)

ssspy_tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ssspy_tests_dir)

from dummy.callback import DummyCallback, dummy_function  # noqa: E402
from dummy.utils.dataset import download_sample_speech_data  # noqa: E402

max_duration = 0.5
n_fft = 512
hop_length = 256
n_bins = n_fft // 2 + 1
n_iter = 3

parameters_callbacks = [None, dummy_function, [DummyCallback(), dummy_function]]
parameters_is_holonomic = [True, False]
parameters_scale_restoration = [True, False, "projection_back", "minimal_distortion_principle"]
parameters_spatial_algorithm = ["IP", "IP1", "IP2"]
parameters_grad_fdica = [
    (2, {}),
    (
        3,
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
]
parameters_aux_fdica = [
    (2, {}),
    (
        3,
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
]


@pytest.mark.parametrize("callbacks", parameters_callbacks)
def test_grad_fdica_base(
    callbacks: Optional[Union[Callable[[GradFDICA], None], List[Callable[[GradFDICA], None]]]],
):
    np.random.seed(111)

    def contrast_fn(y):
        return 2 * np.abs(y)

    def score_fn(y):
        denominator = np.maximum(np.abs(y), 1e-12)
        return y / denominator

    fdica = GradFDICABase(contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks)

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

    waveform_src_img, _ = download_sample_speech_data(
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_duration=max_duration,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    def contrast_fn(y):
        return 2 * np.abs(y)

    def score_fn(y):
        denominator = np.maximum(np.abs(y), 1e-12)
        return y / denominator

    fdica = GradFDICA(
        contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks, is_holonomic=is_holonomic
    )
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape
    assert type(fdica.loss[-1]) is float

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

    waveform_src_img, _ = download_sample_speech_data(
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_duration=max_duration,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    def contrast_fn(y):
        return 2 * np.abs(y)

    def score_fn(y):
        denominator = np.maximum(np.abs(y), 1e-12)
        return y / denominator

    fdica = NaturalGradFDICA(
        contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks, is_holonomic=is_holonomic
    )
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape
    assert type(fdica.loss[-1]) is float

    print(fdica)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_aux_fdica)
@pytest.mark.parametrize("spatial_algorithm", parameters_spatial_algorithm)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("scale_restoration", parameters_scale_restoration)
def test_aux_fdica(
    n_sources: int,
    spatial_algorithm: str,
    callbacks: Optional[Union[Callable[[AuxFDICA], None], List[Callable[[AuxFDICA], None]]]],
    scale_restoration: Union[str, bool],
    reset_kwargs: Dict[Any, Any],
):
    if spatial_algorithm in ["IP"] and not pytest.run_redundant:
        pytest.skip(reason="Need --run-redundant option to run.")

    np.random.seed(111)

    waveform_src_img, _ = download_sample_speech_data(
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_duration=max_duration,
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
        spatial_algorithm=spatial_algorithm,
        contrast_fn=contrast_fn,
        d_contrast_fn=d_contrast_fn,
        callbacks=callbacks,
        scale_restoration=scale_restoration,
    )
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape
    assert type(fdica.loss[-1]) is float

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

    waveform_src_img, _ = download_sample_speech_data(
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_duration=max_duration,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    fdica = GradLaplaceFDICA(callbacks=callbacks, is_holonomic=is_holonomic)
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape
    assert type(fdica.loss[-1]) is float

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

    waveform_src_img, _ = download_sample_speech_data(
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_duration=max_duration,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    fdica = NaturalGradLaplaceFDICA(callbacks=callbacks, is_holonomic=is_holonomic)
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape
    assert type(fdica.loss[-1]) is float

    print(fdica)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_aux_fdica)
@pytest.mark.parametrize("spatial_algorithm", parameters_spatial_algorithm)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("scale_restoration", parameters_scale_restoration)
def test_aux_laplace_fdica(
    n_sources: int,
    spatial_algorithm: str,
    callbacks: Optional[
        Union[Callable[[AuxLaplaceFDICA], None], List[Callable[[AuxLaplaceFDICA], None]]]
    ],
    scale_restoration: Union[str, bool],
    reset_kwargs: Dict[Any, Any],
):
    if spatial_algorithm in ["IP"] and not pytest.run_redundant:
        pytest.skip(reason="Need --run-redundant option to run.")

    np.random.seed(111)

    waveform_src_img, _ = download_sample_speech_data(
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_duration=max_duration,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    fdica = AuxLaplaceFDICA(
        spatial_algorithm=spatial_algorithm,
        callbacks=callbacks,
        scale_restoration=scale_restoration,
    )
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape
    assert type(fdica.loss[-1]) is float

    print(fdica)
