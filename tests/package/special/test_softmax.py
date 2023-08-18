from typing import Optional

import numpy as np
import pytest
import scipy.special

from ssspy.special import softmax

parameters_axis = [0, 1, (0, 2), None]


@pytest.mark.parametrize("axis", parameters_axis)
def test_logsumexp(axis: Optional[int]):
    rng = np.random.default_rng(0)

    n_sources, n_channels = 4, 3
    n_frames = 8
    shape = (n_sources, n_frames, n_channels, n_channels)

    X = rng.random(shape)

    Y = softmax(X, axis=axis)
    Y_scipy = scipy.special.softmax(X, axis=axis)

    assert np.allclose(Y, Y_scipy)
