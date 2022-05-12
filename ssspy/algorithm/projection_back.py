from typing import Optional

import numpy as np


def projection_back(
    data_or_filter: np.ndarray,
    reference: Optional[np.ndarray] = None,
    reference_id: Optional[int] = 0,
):
    r"""Projection back technique for scale alignment.

    The implementation is based on [#murata2001approach]_.

    Args:
        data_or_filter (np.ndarray):
            Estimated spectrograms or demixing filters.
        reference (np.ndarray, optional):
            Reference spectrogram.
        reference_id (int, optional):
            Reference microphone index. Default: 0.

    .. [#murata2001approach]
        N. Murata, S. Ikeda, and A. Ziehe,
        "An approach to blind source separation based on \
        temporal structure of speech signals,"
        *Neurocomputing*, vol. 41, no. 1-4, pp. 1-24, 2001.
    """
    if reference is None:
        W = data_or_filter  # (*, n_sources, n_channels)
        scale = np.linalg.inv(W)  # (*, n_channels, n_sources)

        if reference_id is None:
            scale = scale[..., np.newaxis]  # (*, n_channels, n_sources, 1)
            scale = np.rollaxis(scale, -3, 0)  # (n_channels, *, n_sources, 1)
            demix_filter_scaled = W * scale  # (n_channels, *, n_sources, n_channels)
        else:
            scale = scale[..., reference_id, :]  # (*, n_sources)
            demix_filter_scaled = W * scale[..., np.newaxis]  # (*, n_sources, n_channels)

        return demix_filter_scaled
    else:
        Y = data_or_filter  # (n_sources, n_bins, n_frames)
        X = reference  # (n_channels, n_bins, n_frames)

        Y = Y.transpose(1, 0, 2)  # (n_bins, n_sources, n_frames)
        X = X.transpose(1, 0, 2)  # (n_bins, n_channels, n_frames)
        Y_Hermite = Y.transpose(0, 2, 1).conj()  # (n_bins, n_frames, n_sources)
        XY_Hermite = X @ Y_Hermite  # (n_bins, n_channels, n_sources)
        YY_Hermite = Y @ Y_Hermite  # (n_bins, n_sources, n_sources)

        scale = XY_Hermite @ np.linalg.inv(YY_Hermite)  # (n_bins, n_channels, n_sources)

        if reference_id is None:
            scale = scale.transpose(1, 0, 2)  # (n_channels, n_bins, n_sources)
            Y_scaled = Y * scale[..., np.newaxis]  # (n_channels, n_bins, n_sources, n_frames)
            output_scaled = Y_scaled.swapaxes(-3, -2)  # (n_channels, n_sources, n_bins, n_frames)
            output_scaled.swapaxes
        else:
            scale = scale[..., reference_id, :]  # (n_bins, n_sources)
            Y_scaled = Y * scale[..., np.newaxis]  # (n_bins, n_sources, n_frames)
            output_scaled = Y_scaled.swapaxes(-3, -2)  # (n_sources, n_bins, n_frames)

        return output_scaled
