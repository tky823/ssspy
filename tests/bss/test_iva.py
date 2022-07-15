from typing import Optional, Union, Callable, List, Dict, Any

import pytest
import numpy as np
import scipy.signal as ss

from ssspy.bss.iva import (
    GradIVA,
    NaturalGradIVA,
    FasterIVA,
    AuxIVA,
    GradLaplaceIVA,
    GradGaussIVA,
    NaturalGradLaplaceIVA,
    NaturalGradGaussIVA,
    AuxLaplaceIVA,
    AuxGaussIVA,
)
from ssspy.utils.dataset import download_dummy_data

max_samples = 8000
n_fft = 512
hop_length = 256
n_bins = n_fft // 2 + 1
n_iter = 5


def dummy_function(_) -> None:
    pass


class DummyCallback:
    def __init__(self) -> None:
        pass

    def __call__(self, _) -> None:
        pass


parameters_grad_iva = [
    (2, None, True, {}),
    (
        3,
        dummy_function,
        False,
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
    (2, [DummyCallback(), dummy_function], False, {"demix_filter": None}),
]

parameters_fast_iva = [
    (2, "dev1_female3", None, {}),
    (
        3,
        "dev1_female3",
        dummy_function,
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
    (2, "dev1_female3", [DummyCallback(), dummy_function], {"demix_filter": None}),
]

parameters_aux_iva = [
    (2, "dev1_female3", "IP", None, {}),
    (
        3,
        "dev1_female3",
        "IP2",
        dummy_function,
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
    (2, "dev1_female3", "IP1", [DummyCallback(), dummy_function], {"demix_filter": None}),
    (2, "dev1_female3", "ISS", None, {}),
    (
        3,
        "dev1_female3",
        "ISS1",
        dummy_function,
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
    (4, "dev1_female4", "ISS2", [DummyCallback(), dummy_function], {"demix_filter": None}),
]


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic, reset_kwargs", parameters_grad_iva)
def test_grad_iva(
    n_sources: int,
    callbacks: Optional[Union[Callable[[GradIVA], None], List[Callable[[GradIVA], None]]]],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_dummy_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
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


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic, reset_kwargs", parameters_grad_iva)
def test_natural_grad_iva(
    n_sources: int,
    callbacks: Optional[
        Union[Callable[[NaturalGradIVA], None], List[Callable[[NaturalGradIVA], None]]]
    ],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_dummy_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
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


@pytest.mark.parametrize(
    "n_sources, sisec2010_tag, algorithm_spatial, callbacks, reset_kwargs", parameters_aux_iva
)
def test_aux_iva(
    n_sources: int,
    sisec2010_tag: str,
    algorithm_spatial: str,
    callbacks: Optional[Union[Callable[[AuxIVA], None], List[Callable[[AuxIVA], None]]]],
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_dummy_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_samples=max_samples,
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
        algorithm_spatial=algorithm_spatial,
        contrast_fn=contrast_fn,
        d_contrast_fn=d_contrast_fn,
        callbacks=callbacks,
    )
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize("n_sources, sisec2010_tag, callbacks, reset_kwargs", parameters_fast_iva)
def test_faster_iva(
    n_sources: int,
    sisec2010_tag: str,
    callbacks: Optional[Union[Callable[[AuxIVA], None], List[Callable[[AuxIVA], None]]]],
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_dummy_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_samples=max_samples,
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


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic, reset_kwargs", parameters_grad_iva)
def test_grad_laplace_iva(
    n_sources: int,
    callbacks: Optional[
        Union[Callable[[GradLaplaceIVA], None], List[Callable[[GradLaplaceIVA], None]]]
    ],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_dummy_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    iva = GradLaplaceIVA(callbacks=callbacks, is_holonomic=is_holonomic)
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic, reset_kwargs", parameters_grad_iva)
def test_grad_gauss_iva(
    n_sources: int,
    callbacks: Optional[
        Union[Callable[[GradGaussIVA], None], List[Callable[[GradGaussIVA], None]]]
    ],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_dummy_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    iva = GradGaussIVA(callbacks=callbacks, is_holonomic=is_holonomic)
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic, reset_kwargs", parameters_grad_iva)
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

    waveform_src_img = download_dummy_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    iva = NaturalGradLaplaceIVA(callbacks=callbacks, is_holonomic=is_holonomic)
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic, reset_kwargs", parameters_grad_iva)
def test_natural_grad_gauss_iva(
    n_sources: int,
    callbacks: Optional[
        Union[Callable[[NaturalGradGaussIVA], None], List[Callable[[NaturalGradGaussIVA], None]]]
    ],
    is_holonomic: bool,
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_dummy_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_samples=max_samples,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    iva = NaturalGradGaussIVA(callbacks=callbacks, is_holonomic=is_holonomic)
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize(
    "n_sources, sisec2010_tag, algorithm_spatial, callbacks, reset_kwargs", parameters_aux_iva
)
def test_aux_laplace_iva(
    n_sources: int,
    sisec2010_tag: str,
    algorithm_spatial: str,
    callbacks: Optional[
        Union[
            Callable[[NaturalGradLaplaceIVA], None], List[Callable[[NaturalGradLaplaceIVA], None]]
        ]
    ],
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_dummy_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_samples=max_samples,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    iva = AuxLaplaceIVA(algorithm_spatial=algorithm_spatial, callbacks=callbacks)
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)


@pytest.mark.parametrize(
    "n_sources, sisec2010_tag, algorithm_spatial, callbacks, reset_kwargs", parameters_aux_iva
)
def test_aux_gauss_iva(
    n_sources: int,
    sisec2010_tag: str,
    algorithm_spatial: str,
    callbacks: Optional[
        Union[Callable[[NaturalGradGaussIVA], None], List[Callable[[NaturalGradGaussIVA], None]]]
    ],
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img = download_dummy_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_samples=max_samples,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    iva = AuxGaussIVA(algorithm_spatial=algorithm_spatial, callbacks=callbacks)
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(iva)
