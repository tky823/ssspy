from typing import Optional

import numpy as np


def projection_back(
    data_or_filter: np.ndarray,
    reference: Optional[np.ndarray] = None,
    reference_id: Optional[int] = 0,
):
    if reference is None:
        W = data_or_filter  # (*, n_sources, n_channels)
        scale = np.linalg.inv(W)  # (*, n_channels, n_sources)

        if reference_id is None:
            scale = scale[..., np.newaxis]  # (*, n_channels, n_sources, 1)
            scale = np.rollaxis(scale, -3, 0)  # (n_channels, *, n_sources, 1)
            W_scaled = W * scale  # (n_channels, *, n_sources, n_channels)
        else:
            scale = scale[..., reference_id, :]  # (*, n_sources)
            W_scaled = W * scale[..., np.newaxis]  # (*, n_sources, n_channels)
    else:
        raise NotImplementedError

    return W_scaled
