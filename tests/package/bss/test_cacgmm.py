from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pytest
import scipy.signal as ss
from dummy.callback import DummyCallback, dummy_function
from dummy.utils.dataset import download_sample_speech_data

from ssspy.bss.cacgmm import CACGMM

max_duration = 0.5
window = "hann"
n_fft = 512
hop_length = 256
n_bins = n_fft // 2 + 1
n_iter = 3
rng = np.random.default_rng(42)

parameters_callbacks = [None, dummy_function, [DummyCallback(), dummy_function]]
parameters_permutation_alignment = [
    "posterior_score",
    "amplitude_score",
    "amplitude_correlation",
]
parameters_cacgmm = [(2, 2, {}), (3, 2, {})]


@pytest.mark.parametrize("n_sources, n_channels, reset_kwargs", parameters_cacgmm)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("permutation_alignment", parameters_permutation_alignment)
def test_cacgmm(
    n_sources: int,
    n_channels: int,
    callbacks: Optional[Union[Callable[[CACGMM], None], List[Callable[[CACGMM], None]]]],
    permutation_alignment: bool,
    reset_kwargs: Dict[str, Any],
):
    if n_sources < 4:
        sisec2010_tag = "dev1_female3"
    elif n_sources == 4:
        sisec2010_tag = "dev1_female4"
    else:
        raise ValueError("n_sources should be less than 5.")

    waveform_src_img, _ = download_sample_speech_data(
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_duration=max_duration,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)
    waveform_mix = waveform_mix[:n_channels]

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window=window, nperseg=n_fft, noverlap=n_fft - hop_length
    )

    cacgmm = CACGMM(
        n_sources=n_sources,
        callbacks=callbacks,
        permutation_alignment=permutation_alignment,
        rng=rng,
    )

    spectrogram_est = cacgmm(spectrogram_mix, n_iter=n_iter, **reset_kwargs)

    assert spectrogram_est.shape == (n_sources,) + spectrogram_mix.shape[-2:]
    assert type(cacgmm.loss[-1]) is float

    # when posterior is not given
    _spectrogram_est = cacgmm.separate(spectrogram_mix)

    assert np.allclose(_spectrogram_est, spectrogram_est)

    print(cacgmm)


def test_cacgmm_zero_norm() -> None:
    """Test input with zero norm."""
    n_channels, n_sources, n_samples = 2, 3, 10 * 8000
    waveform_src_img = rng.standard_normal((n_channels, n_sources, n_samples))
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)
    waveform_mix = waveform_mix[:n_channels]

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window=window, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    # set 0 at most grids in 0th frequency bin
    spectrogram_mix[:, 0, 1:-1] = 0

    assert np.linalg.norm(spectrogram_mix, axis=0).any()

    cacgmm = CACGMM(n_sources=n_sources, rng=rng)
    spectrogram_est = cacgmm(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_est.shape == (n_sources,) + spectrogram_mix.shape[-2:]
    assert type(cacgmm.loss[-1]) is float

    # when posterior is not given
    _spectrogram_est = cacgmm.separate(spectrogram_mix)

    assert np.allclose(_spectrogram_est, spectrogram_est)
