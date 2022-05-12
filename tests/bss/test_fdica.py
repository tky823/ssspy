from typing import Optional, Union, Callable, List

import pytest
import numpy as np
import scipy.signal as ss

from ssspy.bss.fdica import GradFDICA, NaturalGradFDICA
from tests.bss.create_dataset import (
    create_sisec2011_dataset,
    create_mird_dataset,
)

max_samples = 16000
n_iter = 5


def dummy_function(_) -> None:
    pass


class DummyCallback:
    def __init__(self) -> None:
        pass

    def __call__(self, _) -> None:
        pass


parameters_grad_fdica = [
    (2, None, True),
    (3, dummy_function, False),
    (2, [DummyCallback(), dummy_function], False),
]


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic", parameters_grad_fdica)
def test_grad_fdica(
    n_sources: str,
    callbacks: Optional[Union[Callable[[GradFDICA], None], List[Callable[[GradFDICA], None]]]],
    is_holonomic: bool,
):
    np.random.seed(111)

    sisec2011_npz_path = create_sisec2011_dataset(n_sources=n_sources)
    mird_npz_path = create_mird_dataset(n_sources=n_sources)

    sisec2011_npz = np.load(sisec2011_npz_path)
    mird_npz = np.load(mird_npz_path)

    waveform_src_img = []

    for src_idx in range(n_sources):
        key = "src_{}".format(src_idx + 1)
        waveform_src = sisec2011_npz[key][:max_samples]
        n_samples = len(waveform_src)
        _waveform_src_img = []

        for waveform_rir in mird_npz[key]:
            waveform_conv = np.convolve(waveform_src, waveform_rir)[:n_samples]
            _waveform_src_img.append(waveform_conv)

        _waveform_src_img = np.stack(_waveform_src_img, axis=0)  # (n_channels, n_samples)
        waveform_src_img.append(_waveform_src_img)

    waveform_src_img = np.stack(waveform_src_img, axis=1)  # (n_channels, n_sources, n_samples)
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(waveform_mix, window="hann", nperseg=2048, noverlap=1024)

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

    print(fdica)


@pytest.mark.parametrize("n_sources, callbacks, is_holonomic", parameters_grad_fdica)
def test_natural_grad_fdica(
    n_sources: str,
    callbacks: Optional[Union[Callable[[GradFDICA], None], List[Callable[[GradFDICA], None]]]],
    is_holonomic: bool,
):
    np.random.seed(111)

    sisec2011_npz_path = create_sisec2011_dataset(n_sources=n_sources)
    mird_npz_path = create_mird_dataset(n_sources=n_sources)

    sisec2011_npz = np.load(sisec2011_npz_path)
    mird_npz = np.load(mird_npz_path)

    waveform_src_img = []

    for src_idx in range(n_sources):
        key = "src_{}".format(src_idx + 1)
        waveform_src = sisec2011_npz[key][:max_samples]
        n_samples = len(waveform_src)
        _waveform_src_img = []

        for waveform_rir in mird_npz[key]:
            waveform_conv = np.convolve(waveform_src, waveform_rir)[:n_samples]
            _waveform_src_img.append(waveform_conv)

        _waveform_src_img = np.stack(_waveform_src_img, axis=0)  # (n_channels, n_samples)
        waveform_src_img.append(_waveform_src_img)

    waveform_src_img = np.stack(waveform_src_img, axis=1)  # (n_channels, n_sources, n_samples)
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(waveform_mix, window="hann", nperseg=2048, noverlap=1024)

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

    print(fdica)
