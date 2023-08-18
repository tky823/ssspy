from typing import Callable, List, Optional, Union

import pytest
from dummy.callback import DummyCallback, dummy_function

from ssspy.bss.base import IterativeMethodBase

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
