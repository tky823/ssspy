import numpy as np
import pytest

from ssspy.bss._solve_permutation import correlation_based_permutation_solver

rng = np.random.default_rng(0)

parameters_give_demixing_filter = [True, False]


@pytest.mark.parametrize("give_demixing_filter", parameters_give_demixing_filter)
def test_correlation_based_permutation_solver(give_demixing_filter: bool):
    n_sources = 3
    n_channels = n_sources
    n_bins, n_frames = 4, 16

    shape = (n_channels, n_bins, n_frames)
    mixture = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    shape = (n_bins, n_sources, n_channels)
    demix_filter = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    separated = demix_filter @ mixture.transpose(1, 0, 2)
    separated = separated.transpose(1, 0, 2)

    assert demix_filter.shape == (n_bins, n_sources, n_channels)
    assert separated.shape == (n_sources, n_bins, n_frames)

    with pytest.warns(UserWarning) as record:
        if give_demixing_filter:
            demix_filter = correlation_based_permutation_solver(
                separated, demix_filter=demix_filter
            )
            assert demix_filter.shape == (n_bins, n_sources, n_channels)
        else:
            separated = correlation_based_permutation_solver(separated)

            assert separated.shape == (n_sources, n_bins, n_frames)

    assert len(record) == 1
    assert (
        str(record[0].message)
        == "Use ssspy.algorithm.permutation_alignment.correlation_based_permutation_solver instead."
    )
