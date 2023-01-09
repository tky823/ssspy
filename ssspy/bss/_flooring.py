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


def add_diagonal(input: np.ndarray, eps: float = EPS) -> np.ndarray:
    r"""Add epsilon to main diagonals of matrix."""
    assert input.shape[-2] == input.shape[-1], "input should be square matrix."

    return input + eps * np.eye(input.shape[-2], input.shape[-1])
