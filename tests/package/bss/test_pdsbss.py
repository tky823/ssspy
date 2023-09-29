from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pytest
import scipy.signal as ss
from dummy.callback import DummyCallback, dummy_function
from dummy.utils.dataset import download_sample_speech_data

from ssspy.bss.pdsbss import PDSBSS, PDSBSSBase

max_duration = 0.5
n_fft = 2048
hop_length = 1024
n_bins = n_fft // 2 + 1
n_iter = 5

parameters_pdsbss = [
    (2, None, {}),
    (
        3,
        dummy_function,
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
    (2, [DummyCallback(), dummy_function], {}),
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
    return y * np.maximum(1 - step_size / norm, 0)


def test_pds_base():
    pdsbss = PDSBSSBase(penalty_fn=penalty_fn, prox_penalty=prox_penalty)

    print(pdsbss)


@pytest.mark.parametrize("n_sources, callbacks, reset_kwargs", parameters_pdsbss)
def test_pdsbss(
    n_sources: int,
    callbacks: Optional[Union[Callable[[PDSBSS], None], List[Callable[[PDSBSS], None]]]],
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

    pdsbss = PDSBSS(penalty_fn=penalty_fn, prox_penalty=prox_penalty, callbacks=callbacks)
    spectrogram_mix_normalized = pdsbss.normalize_by_spectral_norm(spectrogram_mix)
    spectrogram_est = pdsbss(spectrogram_mix_normalized, n_iter=n_iter, **reset_kwargs)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(pdsbss)
