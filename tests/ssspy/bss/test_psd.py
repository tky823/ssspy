from typing import Tuple

import numpy as np
import pytest

from ssspy.bss._psd import to_psd

rng = np.random.default_rng(42)

parameters_shape = [(5, 2, 2), (3, 3)]


@pytest.mark.parametrize("shape", parameters_shape)
def test_to_psd_real(shape: Tuple[int]):
    X = rng.standard_normal(shape)
    X = X @ X.swapaxes(-1, -2)
    X = to_psd(X)
    eigvals = np.linalg.eigvalsh(X)

    assert np.all(X == X.swapaxes(-1, -2))
    assert np.min(eigvals) > 0


@pytest.mark.parametrize("shape", parameters_shape)
def test_to_psd_complex(shape: Tuple[int]):
    X = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    X = X @ X.swapaxes(-1, -2).conj()
    X = to_psd(X)
    eigvals = np.linalg.eigvalsh(X)

    assert np.all(X == X.swapaxes(-1, -2).conj())
    assert np.min(eigvals) > 0
