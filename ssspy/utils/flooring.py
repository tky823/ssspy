from typing import Any, Callable, Optional, Union

import numpy as np

from ..special.flooring import identity


def choose_flooring_fn(
    method: Any,
    flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
) -> Callable[[np.ndarray], np.ndarray]:
    if flooring_fn is None:
        flooring_fn = identity
    elif type(flooring_fn) is str and flooring_fn == "self":
        if hasattr(method, "flooring_fn"):
            flooring_fn = method.flooring_fn
        else:
            flooring_fn = identity

    assert callable(flooring_fn), "flooring_fn should be callable."

    return flooring_fn
