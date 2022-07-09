from typing import Optional, Union, Callable, List, Dict, Any

import pytest
import numpy as np
import scipy.signal as ss

from ssspy.utils.dataset import download_dummy_data
from ssspy.bss.pdsbss import PDSBSS

max_samples = 16000
n_fft = 2048
hop_length = 1024
n_bins = n_fft // 2 + 1
n_iter = 5


def dummy_function(_) -> None:
    pass


class DummyCallback:
    def __init__(self) -> None:
        pass

    def __call__(self, _) -> None:
        pass


parameters_pdsbss = [
    (2, None, {}),
    (
        3,
        dummy_function,
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
    (2, [DummyCallback(), dummy_function], {}),
]


@pytest.mark.parametrize("n_sources, callbacks, reset_kwargs", parameters_pdsbss)
def test_pdsbss(
    n_sources: int,
    callbacks: Optional[Union[Callable[[PDSBSS], None], List[Callable[[PDSBSS], None]]]],
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
            np.ndarray:
                The shape is (n_sources, n_bins, n_frames).
        """
        norm = np.linalg.norm(y, axis=1, keepdims=True)
        return y * np.maximum(1 - step_size / norm, 0)

    pdsbss = PDSBSS(penalty_fn=penalty_fn, prox_penalty=prox_penalty, callbacks=callbacks)
    spectrogram_est = pdsbss(spectrogram_mix, n_iter=n_iter, **reset_kwargs)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(pdsbss)
