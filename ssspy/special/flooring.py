import numpy as np

EPS = 1e-10


def identity(input: np.ndarray) -> np.ndarray:
    r"""Identity function."""
    return input


def max_flooring(input: np.ndarray, eps: float = EPS) -> np.ndarray:
    r"""Max flooring operation."""
    return np.maximum(input, eps)


def add_flooring(input: np.ndarray, eps: float = EPS) -> np.ndarray:
    r"""Add flooring operation."""
    return input + eps
