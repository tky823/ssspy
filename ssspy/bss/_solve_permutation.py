import functools
import warnings
from typing import Callable, Optional

import numpy as np

from ..algorithm.permutation_alignment import (
    correlation_based_permutation_solver as correlation_based_permutation_solver_base,
)
from ..special.flooring import max_flooring

EPS = 1e-10


def correlation_based_permutation_solver(
    separated: np.ndarray,
    demix_filter: Optional[np.ndarray] = None,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
    overwrite: bool = True,
) -> np.ndarray:
    r"""Solve permutaion of estimated spectrograms."""

    warnings.warn(
        "Use ssspy.algorithm.permutation_alignment.correlation_based_permutation_solver instead."
    )

    return correlation_based_permutation_solver_base(
        separated, demix_filter=demix_filter, flooring_fn=flooring_fn, overwrite=overwrite
    )
