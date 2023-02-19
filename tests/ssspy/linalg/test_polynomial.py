import numpy as np

from ssspy.linalg.polynomial import _find_cubic_roots, solve_cubic


def test_find_cubic_roots():
    rng = np.random.default_rng(0)

    n_bins, n_channels = 3, 2

    P = rng.standard_normal((n_bins, n_channels))
    Q = rng.standard_normal((n_bins, n_channels))
    X = _find_cubic_roots(P, Q)
    Y = X**3 + P * X + Q

    assert np.allclose(Y, 0)


def test_solve_cubic():
    rng = np.random.default_rng(0)

    n_bins, n_channels = 3, 2

    # real coefficients
    A = rng.standard_normal((n_bins, n_channels))
    B = rng.standard_normal((n_bins, n_channels))
    C = rng.standard_normal((n_bins, n_channels))
    D = rng.standard_normal((n_bins, n_channels))

    X = solve_cubic(A, B, C)
    Y = X**3 + A * X**2 + B * X + C

    assert np.allclose(Y, 0)

    X = solve_cubic(A, B, C, D)
    Y = A * X**3 + B * X**2 + C * X + D

    assert np.allclose(Y, 0)

    # corner case
    A = np.zeros_like(C)
    B = np.zeros_like(C)

    X = solve_cubic(A, B, C)
    Y = X**3 + A * X**2 + B * X + C

    assert np.allclose(Y, 0)

    # complex coefficients
    A = rng.standard_normal((n_bins, n_channels)) + 1j * rng.standard_normal((n_bins, n_channels))
    B = rng.standard_normal((n_bins, n_channels)) + 1j * rng.standard_normal((n_bins, n_channels))
    C = rng.standard_normal((n_bins, n_channels)) + 1j * rng.standard_normal((n_bins, n_channels))
    D = rng.standard_normal((n_bins, n_channels)) + 1j * rng.standard_normal((n_bins, n_channels))

    X = solve_cubic(A, B, C)
    Y = X**3 + A * X**2 + B * X + C

    assert np.allclose(Y, 0)

    X = solve_cubic(A, B, C, D)
    Y = A * X**3 + B * X**2 + C * X + D

    assert np.allclose(Y, 0)

    # corner case
    A = np.zeros_like(C)
    B = np.zeros_like(C)

    X = solve_cubic(A, B, C)
    Y = X**3 + A * X**2 + B * X + C

    assert np.allclose(Y, 0)
