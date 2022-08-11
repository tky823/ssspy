import pytest

from ssspy.bss._select_pair import sequential_pair_selector, combination_pair_selector

parameters_n_sources = [2, 3, 4]
parameters_ascend = [True, False]


@pytest.mark.parametrize("n_sources", parameters_n_sources)
@pytest.mark.parametrize("ascend", parameters_ascend)
def test_sequential_pair_selector(n_sources: int, ascend: bool):
    for m, n in sequential_pair_selector(n_sources, sort=ascend):
        if ascend:
            assert m < n


@pytest.mark.parametrize("n_sources", parameters_n_sources)
@pytest.mark.parametrize("ascend", parameters_ascend)
def test_combination_pair_selector(n_sources: int, ascend: bool):
    for m, n in combination_pair_selector(n_sources, sort=ascend):
        if ascend:
            assert m < n
