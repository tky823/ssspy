import warnings

import numpy as np

EPS = 1e-10


def identity(input: np.ndarray) -> np.ndarray:
    r"""Identity function."""
    warnings.warn("Use ssspy.special.identity instead.", FutureWarning)

    return input


def max_flooring(input: np.ndarray, eps: float = EPS) -> np.ndarray:
    r"""Max flooring operation."""
    warnings.warn("Use ssspy.special.max_flooring instead.", FutureWarning)

    return np.maximum(input, eps)


def add_flooring(input: np.ndarray, eps: float = EPS) -> np.ndarray:
    r"""Add flooring operation."""
    warnings.warn("Use ssspy.special.add_flooring instead.", FutureWarning)

    return input + eps
