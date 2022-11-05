import os
import sys

import numpy as np
import pytest
import scipy.signal as ss

from ssspy.bss.ipsdta import IPSDTAbase

ssspy_tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ssspy_tests_dir)

from dummy.utils.dataset import download_sample_speech_data  # noqa: E402

rng = np.random.default_rng(42)

parameters_ipsdta_base = [2, 3]


@pytest.mark.parametrize(
    "n_basis",
    parameters_ipsdta_base,
)
def test_ipsdta_base_normalize(
    n_basis: int,
):
    waveform_src_img, _ = download_sample_speech_data(
        n_sources=3,
        sisec2010_tag="dev1_female3",
        max_duration=0.05,
        conv=True,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(waveform_mix, window="hann", nperseg=64, noverlap=32)

    ipsdta = IPSDTAbase(n_basis, rng=rng, record_loss=False)
    _ = ipsdta(spectrogram_mix, n_iter=0, normalization=True)

    U, V = ipsdta.basis.copy(), ipsdta.activation.copy()
    R_old = ipsdta.reconstruct_psdtf(U, V)

    ipsdta.normalize()

    U, V = ipsdta.basis.copy(), ipsdta.activation.copy()
    R = ipsdta.reconstruct_psdtf(U, V)

    assert np.allclose(R, R_old)
