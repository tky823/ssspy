from typing import Tuple

import numpy as np
import pytest

from ssspy.bss._psd import to_psd
from ssspy.special import add_flooring

rng = np.random.default_rng(42)

parameters_shape = [(5, 2, 2), (3, 3)]
parameters_kwargs = [{}, {"flooring_fn": None}, {"flooring_fn": add_flooring}]


@pytest.mark.parametrize("shape", parameters_shape)
@pytest.mark.parametrize("kwargs", parameters_kwargs)
def test_to_psd_real(shape: Tuple[int], kwargs):
    X = rng.standard_normal(shape)
    X = X @ X.swapaxes(-1, -2)
    X = to_psd(X, **kwargs)
    eigvals = np.linalg.eigvalsh(X)

    assert np.all(X == X.swapaxes(-1, -2))
    assert np.min(eigvals) > 0


@pytest.mark.parametrize("shape", parameters_shape)
@pytest.mark.parametrize("kwargs", parameters_kwargs)
def test_to_psd_complex(shape: Tuple[int], kwargs):
    X = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    X = X @ X.swapaxes(-1, -2).conj()
    X = to_psd(X, **kwargs)
    eigvals = np.linalg.eigvalsh(X)

    assert np.all(X == X.swapaxes(-1, -2).conj())
    assert np.min(eigvals) > 0
