import os
import sys
from typing import Callable, List, Optional, Union

import numpy as np
import pytest

from ssspy.bss.ipsdta import BlockDecompositionIPSDTAbase, IPSDTAbase

ssspy_tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ssspy_tests_dir)

from dummy.callback import DummyCallback, dummy_function  # noqa: E402

rng = np.random.default_rng(42)

parameters_callbacks = [None, dummy_function, [DummyCallback(), dummy_function]]
parameters_scale_restoration = [True, False, "projection_back", "minimal_distortion_principle"]
parameters_ipsdta_base = [2]
parameters_block_decomposition_ipsdta_base = [4]


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
