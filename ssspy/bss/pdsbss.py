from typing import Optional, Union, List, Callable
import functools

import numpy as np

from ._flooring import max_flooring
from ..algorithm import projection_back

EPS = 1e-10


class PDSBSSbase:
    def __init__(
        self,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["PDSBSSbase"], None], List[Callable[["PDSBSSbase"], None]]]
        ] = None,
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        if flooring_fn is None:
            self.flooring_fn = lambda x: x
        else:
            self.flooring_fn = flooring_fn

        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None
