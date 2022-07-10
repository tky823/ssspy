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


parameters_gauss_ilrma = [
    (2, 2, "IP", 2, None, True),
    (3, 2, "IP1", 1, dummy_function, "power"),
    (2, 2, "IP", 0.5, [DummyCallback(), dummy_function], "power"),
]


@pytest.mark.parametrize(
    "n_sources, n_basis, algorithm_spatial, domain, callbacks, normalization",
    parameters_gauss_ilrma,
)
def test_gauss_ilrma(
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
        callbacks=callbacks,
        normalization=normalization,
    )
    spectrogram_est = ilrma(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape

    print(ilrma)
