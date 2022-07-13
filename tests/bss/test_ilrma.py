from typing import Optional, Union, Callable, List

import pytest
import numpy as np
import scipy.signal as ss

from ssspy.bss.ilrma import GaussILRMA
from ssspy.utils.dataset import download_dummy_data

max_samples = 16000
n_fft = 2048
hop_length = 1024
n_bins = n_fft // 2 + 1
n_iter = 5
sisec2010_root = "./tests/.data/SiSEC2010"
mird_root = "./tests/.data/MIRD"


def dummy_function(_) -> None:
    pass


class DummyCallback:
    def __init__(self) -> None:
        pass

    def __call__(self, _) -> None:
        pass


parameters_algorithm_spatial = ["IP", "IP1", "ISS", "ISS1"]
parameters_callbacks = [None, dummy_function, [DummyCallback(), dummy_function]]
parameters_gauss_ilrma_latent = [
    (2, 4, 2),
    (3, 3, 1),
]
parameters_gauss_ilrma_wo_latent = [
    (2, 2, 2),
    (3, 1, 1),
]
parameters_normalization_latent = [True, False, "power"]
parameters_normalization_wo_latent = [True, False, "power", "projection_back"]


@pytest.mark.parametrize(
    "n_sources, n_basis, domain", parameters_gauss_ilrma_latent,
)
@pytest.mark.parametrize("algorithm_spatial", parameters_algorithm_spatial)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("normalization", parameters_normalization_latent)
def test_gauss_ilrma_latent(
    n_sources: int,
    n_basis: int,
    algorithm_spatial: str,
    domain: float,
    callbacks: Optional[Union[Callable[[GaussILRMA], None], List[Callable[[GaussILRMA], None]]]],
    normalization: Optional[Union[str, bool]],
):
    np.random.seed(111)

    if n_sources < 4:
        sisec2010_tag = "dev1_female3"
    elif n_sources == 4:
        sisec2010_tag = "dev1_female4"
    else:
        raise ValueError("n_sources should be less than 5.")

    waveform_src_img = download_dummy_data(
        sisec2010_root=sisec2010_root,
        mird_root=mird_root,
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_samples=max_samples,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    ilrma = GaussILRMA(
        n_basis,
        algorithm_spatial=algorithm_spatial,
        domain=domain,
        partitioning=True,
        callbacks=callbacks,
        normalization=normalization,
    )
    spectrogram_est = ilrma(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(ilrma)


@pytest.mark.parametrize(
    "n_sources, n_basis, domain", parameters_gauss_ilrma_wo_latent,
)
@pytest.mark.parametrize("algorithm_spatial", parameters_algorithm_spatial)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("normalization", parameters_normalization_wo_latent)
def test_gauss_ilrma_wo_latent(
    n_sources: int,
    n_basis: int,
    algorithm_spatial: str,
    domain: float,
    callbacks: Optional[Union[Callable[[GaussILRMA], None], List[Callable[[GaussILRMA], None]]]],
    normalization: Optional[Union[str, bool]],
):
    np.random.seed(111)

    if n_sources < 4:
        sisec2010_tag = "dev1_female3"
    elif n_sources == 4:
        sisec2010_tag = "dev1_female4"
    else:
        raise ValueError("n_sources should be less than 5.")

    waveform_src_img = download_dummy_data(
        sisec2010_root=sisec2010_root,
        mird_root=mird_root,
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_samples=max_samples,
    )
    waveform_mix = np.sum(waveform_src_img, axis=1)  # (n_channels, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )

    ilrma = GaussILRMA(
        n_basis,
        algorithm_spatial=algorithm_spatial,
        domain=domain,
        partitioning=False,
        callbacks=callbacks,
        normalization=normalization,
    )
    spectrogram_est = ilrma(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(ilrma)
