import numpy as np
import pytest

from ssspy.bss.ipsdta import BlockDecompositionIPSDTAbase, IPSDTAbase

rng = np.random.default_rng(42)

parameters_ipsdta_base = [2]
parameters_block_decomposition_ipsdta_base = [4]


@pytest.mark.parametrize(
    "n_basis",
    parameters_ipsdta_base,
)
def test_ipsdta_base_normalize(
    n_basis: int,
):
    ipsdta = IPSDTAbase(n_basis, rng=rng, record_loss=False)
    print(ipsdta)


@pytest.mark.parametrize(
    "n_basis",
    parameters_ipsdta_base,
)
@pytest.mark.parametrize(
    "n_blocks",
    parameters_block_decomposition_ipsdta_base,
)
def test_block_decomposition_ipsdta_base(
    n_basis: int,
    n_blocks: int,
):
    ipsdta = BlockDecompositionIPSDTAbase(n_basis, n_blocks, rng=rng, record_loss=False)
    print(ipsdta)
