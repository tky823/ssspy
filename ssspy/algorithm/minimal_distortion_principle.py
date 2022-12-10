from typing import Optional

import numpy as np


def minimal_distortion_principle(
    estimated: np.ndarray,
    reference: Optional[np.ndarray] = None,
    reference_id: Optional[int] = 0,
) -> np.ndarray:
    r"""Minimal distortion principle to restore scale ambiguity.

    The implementation is based on [#matsuoka2002minimal]_.

    Args:
        estimated (numpy.ndarray):
            Estimated spectrograms with shape of (n_channels, n_bins, n_frames).
        reference (numpy.ndarray, optional):
            Reference spectrogram with shape of (n_sources, n_bins, n_frames).
        reference_id (int, optional):
            Reference microphone index. Default: ``0``.

    Returns:
        numpy.ndarray of rescaled estimated spectrograms or demixing filters.

    .. [#matsuoka2002minimal]
        N. Murata, S. Ikeda, and A. Ziehe,
        "Minimal distortion principle for blind source separation,"
        in *Proc. ICA*, 2001, pp. 722-727.
    """
    Y = estimated
    X_conj = reference.conj()

    if reference_id is None:
        num = np.sum(Y * X_conj[:, np.newaxis, :, :], axis=-1, keepdims=True)
    else:
        num = np.sum(Y * X_conj[reference_id], axis=-1, keepdims=True)

    denom = np.sum(np.abs(Y) ** 2, axis=-1, keepdims=True)
    Z = num / denom
    output_scaled = Z.conj() * Y

    return output_scaled
