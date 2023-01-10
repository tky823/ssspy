import os
import sys
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pytest
import scipy.signal as ss

from ssspy.bss.mnmf import GaussMNMF, MNMFbase

ssspy_tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ssspy_tests_dir)

from dummy.callback import DummyCallback, dummy_function  # noqa: E402
from dummy.utils.dataset import download_sample_speech_data  # noqa: E402

max_duration = 0.1
n_fft = 256
hop_length = 128
window = "hann"
n_bins = n_fft // 2 + 1
n_iter = 3
rng = np.random.default_rng(42)


parameters_partitioning = [True, False]
parameters_callbacks = [None, dummy_function, [DummyCallback(), dummy_function]]
parameters_normalization = [True, False]
parameters_mnmf_base = [2]
parameters_mnmf = [
    (2, 2, 2, {}),
    (3, 2, 3, {}),
]


@pytest.mark.parametrize(
    "n_basis",
    parameters_mnmf_base,
)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
def test_mnmf_base(
    n_basis: int,
    callbacks: Optional[Union[Callable[[MNMFbase], None], List[Callable[[MNMFbase], None]]]],
):
    ipsdta = MNMFbase(
        n_basis,
        callbacks=callbacks,
        record_loss=False,
        rng=rng,
    )

    print(ipsdta)


@pytest.mark.parametrize(
    "n_sources, n_channels, n_basis, reset_kwargs",
    parameters_mnmf,
)
@pytest.mark.parametrize(
    "partitioning",
    parameters_partitioning,
)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("normalization", parameters_normalization)
def test_gauss_mnmf(
    n_sources: int,
    n_channels: int,
    n_basis: int,
    partitioning: bool,
    callbacks: Optional[Union[Callable[[GaussMNMF], None], List[Callable[[GaussMNMF], None]]]],
    normalization: Optional[Union[str, bool]],
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
    waveform_mix = np.sum(waveform_src_img[:n_channels], axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window=window, nperseg=n_fft, noverlap=n_fft - hop_length
    )

    mnmf = GaussMNMF(
        n_basis,
        n_sources=n_sources,
        partitioning=partitioning,
        callbacks=callbacks,
        normalization=normalization,
        rng=rng,
    )

    spectrogram_est = mnmf(spectrogram_mix, n_iter=n_iter, **reset_kwargs)

    assert spectrogram_est.shape == (n_sources,) + spectrogram_mix.shape[1:]
    assert type(mnmf.loss[-1]) is float

    print(mnmf)
