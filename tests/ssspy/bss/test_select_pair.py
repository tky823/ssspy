import pytest

from ssspy.utils.pair_selector import combination_pair_selector, sequential_pair_selector

parameters_n_sources = [2, 3, 4]
parameters_step = [1, 2]
parameters_ascend = [True, False]


@pytest.mark.parametrize("n_sources", parameters_n_sources)
@pytest.mark.parametrize("step", parameters_step)
@pytest.mark.parametrize("ascend", parameters_ascend)
def test_sequential_pair_selector(n_sources: int, step: int, ascend: bool):
    for m, n in sequential_pair_selector(n_sources, step=step, sort=ascend):
        if ascend:
            assert m < n


@pytest.mark.parametrize("n_sources", parameters_n_sources)
@pytest.mark.parametrize("ascend", parameters_ascend)
def test_combination_pair_selector(n_sources: int, ascend: bool):
    for m, n in combination_pair_selector(n_sources, sort=ascend):
        if ascend:
            assert m < n
