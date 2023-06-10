import numpy as np

from ssspy.linalg.lqpqm import _find_largest_root


def test_find_largest_root():
    alpha = np.array([-1, 1, 1, -1 + 1j])
    beta = np.array([0, 1, 1, -1 - 1j])
    gamma = np.array([1, 1, 2, 1])

    A = -np.real(alpha + beta + gamma)
    B = np.real(alpha * beta + beta * gamma + gamma * alpha)
    C = -np.real(alpha * beta * gamma)

    X = _find_largest_root(A, B, C)

    assert np.allclose(X, gamma)
