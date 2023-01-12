from typing import Optional

import numpy as np
import pytest
import scipy.special

from ssspy.special import logsumexp

parameters_axis = [0, 1, (0, 2), None]
parameters_keepdims = [True, False]


@pytest.mark.parametrize("axis", parameters_axis)
@pytest.mark.parametrize("keepdims", parameters_keepdims)
def test_logsumexp(axis: Optional[int], keepdims: bool):
    rng = np.random.default_rng(0)

    n_sources, n_channels = 4, 3
    n_frames = 8
    shape = (n_sources, n_frames, n_channels, n_channels)

    X = rng.random(shape)

    Y = logsumexp(X, axis=axis, keepdims=keepdims)
    Y_scipy = scipy.special.logsumexp(X, axis=axis, keepdims=keepdims)

    assert np.allclose(Y, Y_scipy)
