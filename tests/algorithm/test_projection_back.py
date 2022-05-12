from typing import Optional

import pytest
import numpy as np

from ssspy.algorithm import projection_back

parameters = [(2, 0), (3, 2), (2, None)]


@pytest.mark.parametrize("n_sources, reference_id", parameters)
def test_projection_back(n_sources: int, reference_id: Optional[int]):
    np.random.seed(111)

    n_channels = n_sources
    n_bins, n_frames = 17, 10

    spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) + 1j * np.random.randn(
        n_channels, n_bins, n_frames
    )
    demix_filter = np.random.randn(n_bins, n_sources, n_channels) + 1j * np.random.randn(
        n_bins, n_sources, n_channels
    )

    demix_filter_scaled = projection_back(demix_filter, reference_id=reference_id)

    spectrogram_est = demix_filter_scaled @ spectrogram_mix.transpose(1, 0, 2)

    if reference_id is None:
        spectrogram_est = spectrogram_est.transpose(0, 2, 1, 3)

        for _spectrogram_est in spectrogram_est:
            assert spectrogram_mix.shape == _spectrogram_est.shape
    else:
        spectrogram_est = spectrogram_est.transpose(1, 0, 2)

        assert spectrogram_mix.shape == spectrogram_est.shape
