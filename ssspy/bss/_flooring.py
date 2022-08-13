import numpy as np

EPS = 1e-12


def max_flooring(input: np.ndarray, eps=EPS):
    r"""Max flooring operation."""
    return np.maximum(input, eps)


def add_flooring(input: np.ndarray, eps=EPS):
    r"""Add flooring operation."""
    return input + eps
