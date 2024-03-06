import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pytest
import scipy.signal as ss
from dummy.callback import DummyCallback, dummy_function
from dummy.utils.dataset import download_sample_speech_data

from ssspy.bss.admmbss import ADMMBSS, ADMMBSSBase, MaskingADMMBSS

max_duration = 0.5
n_fft = 2048
hop_length = 1024
n_bins = n_fft // 2 + 1
n_iter = 5

parameters_admmbss = [
    (2, None, {}),
    (
        3,
        dummy_function,
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
    (2, [DummyCallback(), dummy_function], {}),
    (
        2,
        None,
        {
            # n_frames=9
            "auxiliary1": np.ones((n_bins, 2, 2), dtype=np.complex128),
            "auxiliary2": np.zeros((1, 2, n_bins, 9), dtype=np.complex128),
            "dual1": np.ones((n_bins, 2, 2), dtype=np.complex128),
            "dual2": np.zeros((1, 2, n_bins, 9), dtype=np.complex128),
        },
    ),
]


def contrast_fn(y: np.ndarray) -> np.ndarray:
    r"""Contrast function.

    Args:
        y (np.ndarray):
            The shape is (n_sources, n_bins, n_frames).

    Returns:
        np.ndarray of the shape is (n_sources, n_frames).
    """
    return 2 * np.linalg.norm(y, axis=1)


def penalty_fn(y: np.ndarray) -> float:
    loss = contrast_fn(y)
    loss = np.sum(loss.mean(axis=-1))
    return loss


def prox_penalty(y: np.ndarray, step_size: float = 1) -> np.ndarray:
    r"""Proximal operator of penalty function.

    Args:
        y (np.ndarray):
            The shape is (n_sources, n_bins, n_frames).
        step_size (float):
            Step size. Default: 1.

    Returns:
        np.ndarray of the shape is (n_sources, n_bins, n_frames).
    """
    norm = np.linalg.norm(y, axis=1, keepdims=True)

    # to suppress warning RuntimeWarning
    norm = np.where(norm < step_size, step_size, norm)

    return y * np.maximum(1 - step_size / norm, 0)


def test_admmbss_base():
    admmbss = ADMMBSSBase(penalty_fn=penalty_fn, prox_penalty=prox_penalty)

    print(admmbss)


@pytest.mark.parametrize("n_sources, callbacks, reset_kwargs", parameters_admmbss)
def test_admmbss(
    n_sources: int,
    callbacks: Optional[Union[Callable[[ADMMBSS], None], List[Callable[[ADMMBSS], None]]]],
    reset_kwargs: Dict[Any, Any],
):
    np.random.seed(111)

    waveform_src_img, _ = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_duration=max_duration,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    admmbss = ADMMBSS(penalty_fn=penalty_fn, prox_penalty=prox_penalty, callbacks=callbacks)
    spectrogram_mix_normalized = admmbss.normalize_by_spectral_norm(spectrogram_mix)
    spectrogram_est = admmbss(spectrogram_mix_normalized, n_iter=n_iter, **reset_kwargs)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(admmbss)


@pytest.mark.parametrize("n_sources, callbacks, reset_kwargs", parameters_admmbss)
def test_masking_admmbss(
    n_sources: int,
    callbacks: Optional[Union[Callable[[ADMMBSS], None], List[Callable[[ADMMBSS], None]]]],
    reset_kwargs: Dict[Any, Any],
) -> None:
    np.random.seed(111)

    waveform_src_img, _ = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag="dev1_female3",
        max_duration=max_duration,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    def hva_mask_fn(y: np.ndarray, mask_iter: int = 2) -> np.ndarray:
        """Masking function to emphasize harmonic components.

        Args:
            y (np.ndarray): The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray of mask. The shape is (n_sources, n_bins, n_frames).
        """
        n_sources, n_bins, _ = y.shape

        gamma = 1 / n_sources

        y = np.maximum(np.abs(y), 1e-10)
        zeta = np.log(y)
        zeta_mean = zeta.mean(axis=1, keepdims=True)
        rho = zeta - zeta_mean
        nu = np.fft.irfft(rho, axis=1, norm="backward")
        nu = nu[:, :n_bins]
        varsigma = np.minimum(1, nu)

        for _ in range(mask_iter):
            varsigma = (1 - np.cos(math.pi * varsigma)) / 2

        xi = np.fft.irfft(varsigma * nu, axis=1, norm="forward")
        xi = xi[:, :n_bins]
        varrho = xi + zeta_mean
        v = np.exp(2 * varrho)
        mask = (v / v.sum(axis=0)) ** gamma

        return mask

    admmbss = MaskingADMMBSS(mask_fn=hva_mask_fn, callbacks=callbacks)
    spectrogram_mix_normalized = admmbss.normalize_by_spectral_norm(spectrogram_mix)

    if "auxiliary2" in reset_kwargs:
        auxiliary2 = reset_kwargs.pop("auxiliary2")

        if auxiliary2.ndim == 4:
            auxiliary2 = auxiliary2.squeeze(axis=0)

        reset_kwargs["auxiliary2"] = auxiliary2

    if "dual2" in reset_kwargs:
        dual2 = reset_kwargs.pop("dual2")

        if dual2.ndim == 4:
            dual2 = dual2.squeeze(axis=0)

        reset_kwargs["dual2"] = dual2

    spectrogram_est = admmbss(spectrogram_mix_normalized, n_iter=n_iter, **reset_kwargs)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(admmbss)
