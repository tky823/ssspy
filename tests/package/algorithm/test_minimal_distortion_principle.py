from typing import Optional

import numpy as np
import pytest

from ssspy.algorithm import minimal_distortion_principle

parameters = [(2, 0), (3, 2), (2, None)]


@pytest.mark.parametrize("n_sources, reference_id", parameters)
def test_minimal_distortion_principle(n_sources: int, reference_id: Optional[int]):
    rng = np.random.default_rng(0)

    n_channels = n_sources
    n_bins, n_frames = 5, 8

    spectrogram_mix = rng.standard_normal(
        (n_channels, n_bins, n_frames)
    ) + 1j * rng.standard_normal((n_channels, n_bins, n_frames))
    demix_filter = rng.standard_normal((n_bins, n_sources, n_channels)) + 1j * rng.standard_normal(
        (n_bins, n_sources, n_channels)
    )
    spectrogram_est = demix_filter @ spectrogram_mix.transpose(1, 0, 2)
    spectrogram_est = spectrogram_est.transpose(1, 0, 2)

    spectrogram_est_scaled = minimal_distortion_principle(
        spectrogram_est, spectrogram_mix, reference_id=reference_id
    )

    if reference_id is None:
        for _spectrogram_est_scaled in spectrogram_est_scaled:
            assert spectrogram_mix.shape == _spectrogram_est_scaled.shape
    else:
        assert spectrogram_mix.shape == spectrogram_est.shape
