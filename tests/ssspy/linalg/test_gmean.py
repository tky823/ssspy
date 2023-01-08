import numpy as np
from scipy.linalg import sqrtm

from ssspy.linalg import gmeanmh


def gmeanmh_scipy(A: np.ndarray, B: np.ndarray, inverse="left") -> np.ndarray:
    def _sqrtm(X) -> np.ndarray:
        return np.stack([sqrtm(x) for x in X], axis=0)

    if inverse == "left":
        AB = np.linalg.solve(A, B)
        G = A @ _sqrtm(AB)
    elif inverse == "right":
        AB = np.linalg.solve(B, A)
        AB = AB.swapaxes(-2, -1).conj()
        G = _sqrtm(AB) @ B
    else:
        raise ValueError(f"Invalid inverse={inverse} is given.")

    return G


def test_gmean():
    rng = np.random.default_rng(0)
    size = (16, 32, 4, 1)

    def create_psd():
        x = rng.random(size) + 1j * rng.random(size)
        XX = x * x.transpose(0, 1, 3, 2).conj()

        return np.mean(XX, axis=0)

    A = create_psd()
    B = create_psd()

    G1 = gmeanmh(A, B)
    G2 = gmeanmh_scipy(A, B, inverse="left")
    G3 = gmeanmh_scipy(A, B, inverse="right")

    assert np.allclose(G1, G2)
    assert np.allclose(G1, G3)
