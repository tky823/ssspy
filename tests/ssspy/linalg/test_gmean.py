import numpy as np
from scipy.linalg import sqrtm

from ssspy.linalg import gmeanmh


def gmeanmh_scipy(A: np.ndarray, B: np.ndarray, inverse="left") -> np.ndarray:
    if inverse == "left":
        AB = np.linalg.solve(A, B)
        G = A @ sqrtm(AB)
    elif inverse == "right":
        AB = np.linalg.solve(B, A)
        AB = AB.swapaxes(-2, -1).conj()
        G = sqrtm(AB) @ B
    else:
        raise ValueError(f"Invalid inverse={inverse} is given.")

    return G


def test_gmean():
    rng = np.random.default_rng(0)
    size = (16, 4, 1)

    def create_psd():
        x = rng.random(size) + 1j * rng.random(size)

        return np.mean(x * x.transpose(0, 2, 1).conj(), axis=0)

    A = create_psd()
    B = create_psd()

    G1 = gmeanmh(A, B)
    G2 = gmeanmh_scipy(A, B, inverse="left")
    G3 = gmeanmh_scipy(A, B, inverse="right")

    assert np.allclose(G1, G2)
    assert np.allclose(G1, G3)
