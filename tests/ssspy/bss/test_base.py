import os
import sys
from typing import Optional, Union, Callable, List

import pytest

from ssspy.bss.base import IterativeMethodBase

ssspy_tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ssspy_tests_dir)

from dummy.callback import DummyCallback, dummy_function  # noqa: E402

n_iter = 3

parameters_callbacks = [None, dummy_function, [DummyCallback(), dummy_function]]
parameters_record_loss = [True, False]


@pytest.mark.parametrize("callbacks", parameters_callbacks)
@pytest.mark.parametrize("record_loss", parameters_record_loss)
def test_iterative_method_base(
    callbacks: Optional[
        Union[Callable[[IterativeMethodBase], None], List[Callable[[IterativeMethodBase], None]]]
    ],
    record_loss: bool,
):
    method = IterativeMethodBase(callbacks=callbacks, record_loss=record_loss)

    with pytest.raises(NotImplementedError) as exc_info:
        method(n_iter=n_iter)

    assert exc_info.type is NotImplementedError
