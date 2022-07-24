from typing import Optional, Callable
import functools

import numpy as np

from ._flooring import max_flooring

EPS = 1e-10


def update_by_ip1(
    demix_filter: np.ndarray,
    weighted_covariance: np.ndarray,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
    overwrite: bool = True,
) -> np.ndarray:
    r"""Update demixing filters by iterative projection.

    Args:
        demix_filter (numpy.ndarray):
            Demixing filters to be updated. \
            The shape is (n_bins, n_sources, n_channels).
        weighted_covariance (numpy.ndarray):
            Weighted covariance matrix. \
            The shape is (n_bins, n_sources, n_channels, n_channels).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used. \
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        overwrite (bool):
            Overwrite ``demix_filter`` if ``overwrite=True``. \
            Default: ``True``.

    Returns:
        numpy.ndarray:
            Updated demixing filters. \
            The shape is (n_bins, n_sources, n_channels).
    """
    if overwrite:
        W = demix_filter
    else:
        W = demix_filter.copy()

    U = weighted_covariance

    n_bins, n_sources, n_channels = W.shape

    E = np.eye(n_sources, n_channels)  # (n_sources, n_channels)
    E = np.tile(E, reps=(n_bins, 1, 1))  # (n_bins, n_sources, n_channels)

    for src_idx in range(n_sources):
        w_n_Hermite = W[:, src_idx, :]  # (n_bins, n_channels)
        U_n = U[:, src_idx, :, :]
        e_n = E[:, src_idx, :]  # (n_bins, n_n_channels)

        WU = W @ U_n
        w_n = np.linalg.solve(WU, e_n)  # (n_bins, n_channels)
        wUw = w_n[:, np.newaxis, :].conj() @ U_n @ w_n[:, :, np.newaxis]
        wUw = np.real(wUw[..., 0])
        wUw = np.maximum(wUw, 0)
        denom = np.sqrt(wUw)
        denom = flooring_fn(denom)
        w_n_Hermite = w_n.conj() / denom
        W[:, src_idx, :] = w_n_Hermite

    return W


def update_by_iss1(
    output: np.ndarray,
    weight: np.ndarray,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
) -> np.ndarray:
    r"""Update estimated spectrogram by iterative source steering.

    Args:
        output (numpy.ndarray):
            Estimated spectrograms to be updated. \
            The shape is (n_sources, n_bins, n_frames).
        weight (numpy.ndarray):
            Weights for estimated spectrogram. \
            The shape is (n_sources, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used. \
            Default: ``functools.partial(max_flooring, eps=1e-10)``.

    Returns:
        numpy.ndarray:
            Updated spectrograms. \
            The shape is (n_sources, n_bins, n_frames).
    """
    Y = output
    varphi = weight

    n_sources = Y.shape[0]

    for src_idx in range(n_sources):
        Y_n = Y[src_idx]  # (n_bins, n_frames)

        YY_n_conj = Y * Y_n.conj()
        YY_n = np.abs(Y_n) ** 2
        num = np.mean(varphi * YY_n_conj, axis=-1)
        denom = np.mean(varphi * YY_n, axis=-1)
        denom = flooring_fn(denom)
        v_n = num / denom
        v_n[src_idx] = 1 - 1 / np.sqrt(denom[src_idx])

        Y = Y - v_n[:, :, np.newaxis] * Y_n

    return Y
