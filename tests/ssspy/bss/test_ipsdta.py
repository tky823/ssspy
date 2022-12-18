import os
import sys
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pytest
import scipy.signal as ss

from ssspy.bss.ipsdta import TIPSDTA, BlockDecompositionIPSDTAbase, GaussIPSDTA, IPSDTAbase

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

parameters_dof = [100]
parameters_spatial_algorithm = ["FPI", "VCD"]
parameters_callbacks = [None, dummy_function, [DummyCallback(), dummy_function]]
parameters_source_normalization = [True, False]
parameters_scale_restoration = [True, False, "projection_back", "minimal_distortion_principle"]
parameters_ipsdta_base = [2]
parameters_block_decomposition_ipsdta_base = [4]
parameters_ipsdta = [
    (
        2,
        2,
        43,
        {
            "demix_filter": np.tile(np.eye(2, dtype=np.complex128), (n_bins, 1, 1)),
        },
    ),
    (3, 2, 64, {}),
]


@pytest.mark.parametrize(
    "n_basis",
    parameters_ipsdta_base,
)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("scale_restoration", parameters_scale_restoration)
def test_ipsdta_base(
    n_basis: int,
    callbacks: Optional[Union[Callable[[IPSDTAbase], None], List[Callable[[IPSDTAbase], None]]]],
    scale_restoration: Union[str, bool],
):
    ipsdta = IPSDTAbase(
        n_basis,
        callbacks=callbacks,
        scale_restoration=scale_restoration,
        record_loss=False,
        rng=rng,
    )

    print(ipsdta)


@pytest.mark.parametrize(
    "n_basis",
    parameters_ipsdta_base,
)
@pytest.mark.parametrize(
    "n_blocks",
    parameters_block_decomposition_ipsdta_base,
)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("scale_restoration", parameters_scale_restoration)
def test_block_decomposition_ipsdta_base(
    n_basis: int,
    n_blocks: int,
    callbacks: Optional[
        Union[
            Callable[[BlockDecompositionIPSDTAbase], None],
            List[Callable[[BlockDecompositionIPSDTAbase], None]],
        ]
    ],
    scale_restoration: Union[str, bool],
):
    ipsdta = BlockDecompositionIPSDTAbase(
        n_basis,
        n_blocks,
        callbacks=callbacks,
        scale_restoration=scale_restoration,
        record_loss=False,
        rng=rng,
    )

    print(ipsdta)


@pytest.mark.parametrize(
    "n_sources, n_basis, n_blocks, reset_kwargs",
    parameters_ipsdta,
)
@pytest.mark.parametrize("spatial_algorithm", parameters_spatial_algorithm)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("source_normalization", parameters_source_normalization)
@pytest.mark.parametrize("scale_restoration", parameters_scale_restoration)
def test_gauss_ipsdta(
    n_sources: int,
    n_basis: int,
    n_blocks: int,
    spatial_algorithm: str,
    callbacks: Optional[Union[Callable[[GaussIPSDTA], None], List[Callable[[GaussIPSDTA], None]]]],
    source_normalization: Optional[Union[str, bool]],
    scale_restoration: Union[str, bool],
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

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window=window, nperseg=n_fft, noverlap=n_fft - hop_length
    )

    ipsdta = GaussIPSDTA(
        n_basis,
        n_blocks,
        spatial_algorithm=spatial_algorithm,
        callbacks=callbacks,
        source_normalization=source_normalization,
        scale_restoration=scale_restoration,
        rng=rng,
    )

    if spatial_algorithm == "FPI":
        with pytest.raises(NotImplementedError) as e:
            spectrogram_est = ipsdta(spectrogram_mix, n_iter=n_iter, **reset_kwargs)

        assert str(e.value) == "IPSDTA with fixed-point iteration is not supported."
    else:
        spectrogram_est = ipsdta(spectrogram_mix, n_iter=n_iter, **reset_kwargs)

        assert spectrogram_mix.shape == spectrogram_est.shape
        assert type(ipsdta.loss[-1]) is float

    print(ipsdta)


@pytest.mark.parametrize(
    "n_sources, n_basis, n_blocks, reset_kwargs",
    parameters_ipsdta,
)
@pytest.mark.parametrize("dof", parameters_dof)
@pytest.mark.parametrize("spatial_algorithm", parameters_spatial_algorithm)
@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("source_normalization", parameters_source_normalization)
@pytest.mark.parametrize("scale_restoration", parameters_scale_restoration)
def test_t_ipsdta(
    n_sources: int,
    n_basis: int,
    n_blocks: int,
    dof: float,
    spatial_algorithm: str,
    callbacks: Optional[Union[Callable[[GaussIPSDTA], None], List[Callable[[GaussIPSDTA], None]]]],
    source_normalization: Optional[Union[str, bool]],
    scale_restoration: Union[str, bool],
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

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, window=window, nperseg=n_fft, noverlap=n_fft - hop_length
    )

    ipsdta = TIPSDTA(
        n_basis,
        n_blocks,
        dof=dof,
        spatial_algorithm=spatial_algorithm,
        callbacks=callbacks,
        source_normalization=source_normalization,
        scale_restoration=scale_restoration,
        rng=rng,
    )

    if spatial_algorithm != "VCD":
        with pytest.raises(NotImplementedError) as e:
            spectrogram_est = ipsdta(spectrogram_mix, n_iter=n_iter, **reset_kwargs)

        assert str(e.value) == "Not support {}.".format(spatial_algorithm)
    else:
        spectrogram_est = ipsdta(spectrogram_mix, n_iter=n_iter, **reset_kwargs)

        assert spectrogram_mix.shape == spectrogram_est.shape
        assert type(ipsdta.loss[-1]) is float

    print(ipsdta)
