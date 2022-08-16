import pytest
import numpy as np

from ssspy.linalg import inv2

parameters_sources = [2, 5]


@pytest.mark.parametrize("n_sources", parameters_sources)
def test_inv2(n_sources: int):
    np.random.seed(111)

    shape = (n_sources, 2, 2)

    A = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    B = inv2(A)

    assert np.allclose(A @ B, np.eye(2))
