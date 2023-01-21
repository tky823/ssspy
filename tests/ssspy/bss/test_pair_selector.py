import pytest

from ssspy.bss._select_pair import combination_pair_selector, sequential_pair_selector

parameters_n_sources = [4]
parameters_step = [1, 2]
parameters_ascend = [True, False]


@pytest.mark.parametrize("n_sources", parameters_n_sources)
@pytest.mark.parametrize("step", parameters_step)
@pytest.mark.parametrize("ascend", parameters_ascend)
def test_sequential_pair_selector(n_sources: int, step: int, ascend: bool):
    with pytest.warns(UserWarning) as record:
        sequential_pair_selector(n_sources, step=step, sort=ascend)

    assert len(record) == 1
    assert str(record[0].message) == "Use ssspy.utils.select_pair.sequential_pair_selector instead."


@pytest.mark.parametrize("n_sources", parameters_n_sources)
@pytest.mark.parametrize("ascend", parameters_ascend)
def test_combination_pair_selector(n_sources: int, ascend: bool):
    with pytest.warns(UserWarning) as record:
        combination_pair_selector(n_sources, sort=ascend)

    assert len(record) == 1
    assert (
        str(record[0].message) == "Use ssspy.utils.select_pair.combination_pair_selector instead."
    )
