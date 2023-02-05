from typing import Any, Callable, Optional, Union

import numpy as np

from ..special.flooring import identity


def choose_flooring_fn(
    flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    method: Optional[Any] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    if flooring_fn is None:
        assert method is None, "method is given, but flooring function is not specified."

        flooring_fn = identity
    elif type(flooring_fn) is str and flooring_fn == "self":
        if method is None or not hasattr(method, "flooring_fn"):
            flooring_fn = identity
        else:
            flooring_fn = method.flooring_fn

    assert callable(flooring_fn), "flooring_fn should be callable."

    return flooring_fn
