from typing import Optional, Union, Callable, List, Dict, Any

import pytest
import numpy as np
import scipy.signal as ss

from ssspy.bss.iva import IVAbase
from ssspy.bss.iva import FastIVAbase, FastIVA, FasterIVA
from ssspy.bss.iva import GradIVAbase, GradIVA, GradLaplaceIVA, GradGaussIVA
from ssspy.bss.iva import NaturalGradIVA, NaturalGradLaplaceIVA, NaturalGradGaussIVA
from ssspy.bss.iva import (
    AuxIVAbase,
    AuxIVA,
    AuxLaplaceIVA,
    AuxGaussIVA,
)
from ssspy.utils.dataset import download_sample_speech_data
from tests.dummy.callback import DummyCallback, dummy_function

max_samples = 8000
n_fft = 512
hop_length = 256
n_bins = n_fft // 2 + 1
n_iter = 3

parameters_spatial_algorithm = ["IP", "IP1", "IP2", "ISS", "ISS1", "ISS2"]
parameters_callbacks = [None, dummy_function, [DummyCallback(), dummy_function]]
parameters_is_holonomic = [True, False]
parameters_scale_restoration = [True, False, "projection_back"]
parameters_grad_iva = [
    (2, {}),
    (3, {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},),
]
parameters_fast_iva = [
    (2, "dev1_female3", {}),
    (
        3,
        "dev1_female3",
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
    (2, "dev1_female3", {"demix_filter": None}),
]
parameters_aux_iva = [
    (2, "dev1_female3", {}),
    (
        3,
        "dev1_female3",
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
    (2, "dev1_female3", {"demix_filter": None}),
    (
        3,
        "dev1_female3",
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
    (4, "dev1_female4", {"demix_filter": None}),
]


@pytest.mark.parametrize("callbacks", parameters_callbacks)
def test_iva_base(
    callbacks: Optional[Union[Callable[[AuxIVA], None], List[Callable[[AuxIVA], None]]]],
):
    iva = IVAbase(callbacks=callbacks)

    print(iva)


@pytest.mark.parametrize("callbacks", parameters_callbacks)
def test_fast_iva_base(
    callbacks: Optional[Union[Callable[[AuxIVA], None], List[Callable[[AuxIVA], None]]]],
):
    np.random.seed(111)

    iva = FastIVAbase(callbacks=callbacks)

    print(iva)


@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_grad_iva_base(
    callbacks: Optional[Union[Callable[[GradIVA], None], List[Callable[[GradIVA], None]]]],
    is_holonomic: bool,
):
    np.random.seed(111)

    def contrast_fn(y: np.ndarray) -> np.ndarray:
        r"""Contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.linalg.norm(y, axis=1)

    def score_fn(y) -> np.ndarray:
        r"""Score function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_bins, n_frames).
        """
        norm = np.linalg.norm(y, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-10)
        return y / norm

    iva = GradIVAbase(
        contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks, is_holonomic=is_holonomic
    )

    print(iva)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_iva)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_grad_iva(
    n_sources: int,
    callbacks: Optional[Union[Callable[[GradIVA], None], List[Callable[[GradIVA], None]]]],
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

    def contrast_fn(y: np.ndarray) -> np.ndarray:
        r"""Contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.linalg.norm(y, axis=1)

    def score_fn(y) -> np.ndarray:
        r"""Score function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_bins, n_frames).
        """
        norm = np.linalg.norm(y, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-10)
        return y / norm

    iva = GradIVA(
        contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks, is_holonomic=is_holonomic
    )
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_iva)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_natural_grad_iva(
    n_sources: int,
    callbacks: Optional[
        Union[Callable[[NaturalGradIVA], None], List[Callable[[NaturalGradIVA], None]]]
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

    def contrast_fn(y: np.ndarray) -> np.ndarray:
        r"""Contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.linalg.norm(y, axis=1)

    def score_fn(y):
        r"""Score function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_bins, n_frames).
        """
        norm = np.linalg.norm(y, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-10)
        return y / norm

    iva = NaturalGradIVA(
        contrast_fn=contrast_fn, score_fn=score_fn, callbacks=callbacks, is_holonomic=is_holonomic
    )
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize("n_sources, sisec2010_tag, reset_kwargs", parameters_fast_iva)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
def test_fast_iva(
    n_sources: int,
    sisec2010_tag: str,
    callbacks: Optional[Union[Callable[[AuxIVA], None], List[Callable[[AuxIVA], None]]]],
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_samples=max_samples,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    def contrast_fn(y: np.ndarray) -> np.ndarray:
        r"""Contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.linalg.norm(y, axis=1)

    def d_contrast_fn(y) -> np.ndarray:
        r"""Derivative of contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.ones_like(y)

    def dd_contrast_fn(y) -> np.ndarray:
        r"""Second order derivative of contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.zeros_like(y)

    iva = FastIVA(
        contrast_fn=contrast_fn,
        d_contrast_fn=d_contrast_fn,
        dd_contrast_fn=dd_contrast_fn,
        callbacks=callbacks,
    )
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize("n_sources, sisec2010_tag, reset_kwargs", parameters_fast_iva)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
def test_faster_iva(
    n_sources: int,
    sisec2010_tag: str,
    callbacks: Optional[Union[Callable[[AuxIVA], None], List[Callable[[AuxIVA], None]]]],
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_samples=max_samples,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    def contrast_fn(y: np.ndarray) -> np.ndarray:
        r"""Contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.linalg.norm(y, axis=1)

    def d_contrast_fn(y) -> np.ndarray:
        r"""Derivative of contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.ones_like(y)

    iva = FasterIVA(contrast_fn=contrast_fn, d_contrast_fn=d_contrast_fn, callbacks=callbacks)
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("scale_restoration", parameters_scale_restoration)
def test_aux_iva_base(
    callbacks: Optional[Union[Callable[[AuxIVA], None], List[Callable[[AuxIVA], None]]]],
    scale_restoration: Union[str, bool],
):
    np.random.seed(111)

    def contrast_fn(y: np.ndarray) -> np.ndarray:
        r"""Contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.linalg.norm(y, axis=1)

    def d_contrast_fn(y) -> np.ndarray:
        r"""Derivative of contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.ones_like(y)

    iva = AuxIVAbase(
        contrast_fn=contrast_fn,
        d_contrast_fn=d_contrast_fn,
        callbacks=callbacks,
        scale_restoration=scale_restoration,
    )

    print(iva)


@pytest.mark.parametrize("n_sources, sisec2010_tag, reset_kwargs", parameters_aux_iva)
@pytest.mark.parametrize("spatial_algorithm", parameters_spatial_algorithm)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("scale_restoration", parameters_scale_restoration)
def test_aux_iva(
    n_sources: int,
    sisec2010_tag: str,
    spatial_algorithm: str,
    callbacks: Optional[Union[Callable[[AuxIVA], None], List[Callable[[AuxIVA], None]]]],
    scale_restoration: Union[str, bool],
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_samples=max_samples,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    def contrast_fn(y: np.ndarray) -> np.ndarray:
        r"""Contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.linalg.norm(y, axis=1)

    def d_contrast_fn(y) -> np.ndarray:
        r"""Derivative of contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.ones_like(y)

    iva = AuxIVA(
        spatial_algorithm=spatial_algorithm,
        contrast_fn=contrast_fn,
        d_contrast_fn=d_contrast_fn,
        callbacks=callbacks,
        scale_restoration=scale_restoration,
    )
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_iva)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_grad_laplace_iva(
    n_sources: int,
    callbacks: Optional[
        Union[Callable[[GradLaplaceIVA], None], List[Callable[[GradLaplaceIVA], None]]]
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

    iva = GradLaplaceIVA(callbacks=callbacks, is_holonomic=is_holonomic)
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_iva)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_grad_gauss_iva(
    n_sources: int,
    callbacks: Optional[
        Union[Callable[[GradGaussIVA], None], List[Callable[[GradGaussIVA], None]]]
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

    iva = GradGaussIVA(callbacks=callbacks, is_holonomic=is_holonomic)
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_iva)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_natural_grad_laplace_iva(
    n_sources: int,
    callbacks: Optional[
        Union[
            Callable[[NaturalGradLaplaceIVA], None], List[Callable[[NaturalGradLaplaceIVA], None]]
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

    iva = NaturalGradLaplaceIVA(callbacks=callbacks, is_holonomic=is_holonomic)
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize("n_sources, reset_kwargs", parameters_grad_iva)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_natural_grad_gauss_iva(
    n_sources: int,
    callbacks: Optional[
        Union[Callable[[NaturalGradGaussIVA], None], List[Callable[[NaturalGradGaussIVA], None]]]
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

    iva = NaturalGradGaussIVA(callbacks=callbacks, is_holonomic=is_holonomic)
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize("n_sources, sisec2010_tag, reset_kwargs", parameters_aux_iva)
@pytest.mark.parametrize("spatial_algorithm", parameters_spatial_algorithm)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("scale_restoration", parameters_scale_restoration)
def test_aux_laplace_iva(
    n_sources: int,
    sisec2010_tag: str,
    spatial_algorithm: str,
    callbacks: Optional[
        Union[
            Callable[[NaturalGradLaplaceIVA], None], List[Callable[[NaturalGradLaplaceIVA], None]]
        ]
    ],
    scale_restoration: Union[str, bool],
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_samples=max_samples,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    iva = AuxLaplaceIVA(
        spatial_algorithm=spatial_algorithm,
        callbacks=callbacks,
        scale_restoration=scale_restoration,
    )
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize("n_sources, sisec2010_tag, reset_kwargs", parameters_aux_iva)
@pytest.mark.parametrize("spatial_algorithm", parameters_spatial_algorithm)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("scale_restoration", parameters_scale_restoration)
def test_aux_gauss_iva(
    n_sources: int,
    sisec2010_tag: str,
    spatial_algorithm: str,
    callbacks: Optional[
        Union[Callable[[NaturalGradGaussIVA], None], List[Callable[[NaturalGradGaussIVA], None]]]
    ],
    scale_restoration: Union[str, bool],
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_samples=max_samples,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    iva = AuxGaussIVA(
        spatial_algorithm=spatial_algorithm,
        callbacks=callbacks,
        scale_restoration=scale_restoration,
    )
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)
