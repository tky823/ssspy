from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pytest
import scipy.signal as ss
from dummy.callback import DummyCallback, dummy_function
from dummy.utils.dataset import download_sample_speech_data

from ssspy.bss.hva import HVA, MaskingPDSHVA

max_duration = 0.5
n_fft = 2048
hop_length = 1024
n_bins = n_fft // 2 + 1
n_iter = 5

parameters_pdshva = [
    (2, None, {}),
    (
        3,
        dummy_function,
        {"demix_filter": np.tile(-np.eye(3, dtype=np.complex128), reps=(n_bins, 1, 1))},
    ),
    (2, [DummyCallback(), dummy_function], {}),
]


@pytest.mark.parametrize("n_sources, callbacks, reset_kwargs", parameters_pdshva)
def test_masking_pdshva(
    n_sources: int,
    callbacks: Optional[
        Union[Callable[[MaskingPDSHVA], None], List[Callable[[MaskingPDSHVA], None]]]
    ],
    reset_kwargs: Dict[Any, Any],
):
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

    hva = MaskingPDSHVA(callbacks=callbacks)

    spectrogram_mix_normalized = hva.normalize_by_spectral_norm(spectrogram_mix)
    spectrogram_est = hva(spectrogram_mix_normalized, n_iter=n_iter, **reset_kwargs)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(hva)


@pytest.mark.parametrize("n_sources, callbacks, reset_kwargs", parameters_pdshva)
def test_hva(
    n_sources: int,
    callbacks: Optional[Union[Callable[[HVA], None], List[Callable[[HVA], None]]]],
    reset_kwargs: Dict[Any, Any],
):
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

    hva = HVA(callbacks=callbacks)

    spectrogram_mix_normalized = hva.normalize_by_spectral_norm(spectrogram_mix)
    spectrogram_est = hva(spectrogram_mix_normalized, n_iter=n_iter, **reset_kwargs)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(hva)
