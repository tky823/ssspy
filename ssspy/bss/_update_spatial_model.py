from typing import Optional, Tuple, Callable, Iterable
import functools

import numpy as np

from ._flooring import identity, max_flooring
from ._select_pair import sequential_pair_selector
from ..linalg import inv2, eigh2

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
    if flooring_fn is None:
        flooring_fn = identity

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


def update_by_ip2(
    demix_filter: np.ndarray,
    weighted_covariance: np.ndarray,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
    pair_selector: Optional[Callable[[int], Iterable[Tuple[int, int]]]] = None,
    overwrite: bool = True,
) -> np.ndarray:
    r"""Update demixing filters by pairwise iterative projection.

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
        pair_selector (callable, optional):
            Selector to choose updaing pair. \
            If ``None`` is given, ``sequential_pair_selector`` is used. \
            Default: ``None``.
        overwrite (bool):
            Overwrite ``demix_filter`` if ``overwrite=True``. \
            Default: ``True``.

    Returns:
        numpy.ndarray:
            Updated demixing filters. \
            The shape is (n_bins, n_sources, n_channels).
    """
    if flooring_fn is None:
        flooring_fn = identity

    if pair_selector is None:
        pair_selector = sequential_pair_selector

    if overwrite:
        W = demix_filter
    else:
        W = demix_filter.copy()

    U = weighted_covariance.transpose(1, 0, 2, 3)

    n_sources = W.shape[1]

    for m, n in pair_selector(n_sources):
        W_mn = W[:, (m, n), :]  # (n_bins, 2, n_channels)
        U_mn = U[(m, n), :, :, :]  # (2, n_bins, n_channels, n_channels)

        G_mn = W_mn @ U_mn @ W_mn.transpose(0, 2, 1).conj()  # (2, n_bins, 2, 2)

        G_m, G_n = G_mn
        _, H_mn = eigh2(G_m, G_n)  # (n_bins, 2, 2)
        h_mn = H_mn.transpose(2, 0, 1)  # (2, n_bins, 2)
        hGh_mn = h_mn[:, :, np.newaxis, :].conj() @ G_mn @ h_mn[:, :, :, np.newaxis]
        hGh_mn = np.squeeze(hGh_mn, axis=-1)  # (2, n_bins, 1)
        hGh_mn = np.real(hGh_mn)
        hGh_mn = np.maximum(hGh_mn, 0)
        denom_mn = np.sqrt(hGh_mn)
        denom_mn = flooring_fn(denom_mn)
        h_mn = h_mn / denom_mn
        H_mn = h_mn.transpose(1, 2, 0)
        W_mn_conj = W_mn.transpose(0, 2, 1).conj() @ H_mn

        W[:, (m, n), :] = W_mn_conj.transpose(0, 2, 1).conj()

    return W


def update_by_iss1(
    separated: np.ndarray,
    weight: np.ndarray,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
) -> np.ndarray:
    r"""Update estimated spectrogram by iterative source steering.

    Args:
        separated (numpy.ndarray):
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
    if flooring_fn is None:
        flooring_fn = identity

    Y = separated
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


def update_by_iss2(
    separated: np.ndarray,
    weight: np.ndarray,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
    pair_selector: Optional[Callable[[int], Iterable[Tuple[int, int]]]] = None,
) -> np.ndarray:
    r"""Update estimated spectrogram by pairwise iterative source steering.

    Args:
        separated (numpy.ndarray):
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
        pair_selector (callable, optional):
            Selector to choose updaing pair. \
            If ``None`` is given, ``sequential_pair_selector`` is used. \
            Default: ``None``.

    Returns:
        numpy.ndarray:
            Updated spectrograms. \
            The shape is (n_sources, n_bins, n_frames).
    """
    Y = separated
    varphi = weight

    n_sources = Y.shape[0]

    if flooring_fn is None:
        flooring_fn = identity

    if pair_selector is None:
        pair_selector = functools.partial(sequential_pair_selector, stop=n_sources, step=2)

    for m, n in pair_selector(n_sources):
        if m < 0:
            m = n_sources + m
        if n < 0:
            n = n_sources + n

        if m > n:
            ascend = False
            m, n = n, m
        else:
            ascend = True

        # Split into main and sub
        Y_1, Y_m, Y_2, Y_n, Y_3 = np.split(Y, [m, m + 1, n, n + 1], axis=0)
        Y_sub = np.concatenate([Y_1, Y_2, Y_3], axis=0)  # (n_sources - 2, n_bins, n_frames)
        varphi_1, varphi_m, varphi_2, varphi_n, varphi_3 = np.split(
            varphi, [m, m + 1, n, n + 1], axis=0
        )
        varphi_sub = np.concatenate([varphi_1, varphi_2, varphi_3], axis=0)

        if ascend:
            Y_main = np.concatenate([Y_m, Y_n], axis=0)  # (2, n_bins, n_frames)
            varphi_main = np.concatenate([varphi_m, varphi_n], axis=0)
        else:
            Y_main = np.concatenate([Y_n, Y_m], axis=0)  # (2, n_bins, n_frames)
            varphi_main = np.concatenate([varphi_n, varphi_m], axis=0)

        YY_main = Y_main[:, np.newaxis, :, :] * Y_main[np.newaxis, :, :, :].conj()
        YY_sub = Y_main[:, np.newaxis, :, :] * Y_sub[np.newaxis, :, :, :].conj()
        YY_main = YY_main.transpose(2, 0, 1, 3)
        YY_sub = YY_sub.transpose(1, 2, 0, 3)

        Y_main = Y_main.transpose(1, 0, 2)

        # Sub
        G_sub = np.mean(
            varphi_sub[:, :, np.newaxis, np.newaxis, :] * YY_main[np.newaxis, :, :, :, :],
            axis=-1,
        )
        F = np.mean(varphi_sub[:, :, np.newaxis, :] * YY_sub, axis=-1)
        Q = -inv2(G_sub) @ F[:, :, :, np.newaxis]
        Q = Q.squeeze(axis=-1)
        Q = Q.transpose(1, 0, 2)
        QY = Q.conj() @ Y_main
        Y_sub = Y_sub + QY.transpose(1, 0, 2)

        # Main
        G_main = np.mean(
            varphi_main[:, :, np.newaxis, np.newaxis, :] * YY_main[np.newaxis, :, :, :, :],
            axis=-1,
        )
        G_m, G_n = G_main
        _, H_mn = eigh2(G_m, G_n)
        h_mn = H_mn.transpose(2, 0, 1)
        hGh_mn = h_mn[:, :, np.newaxis, :].conj() @ G_main @ h_mn[:, :, :, np.newaxis]
        hGh_mn = np.squeeze(hGh_mn, axis=-1)
        hGh_mn = np.real(hGh_mn)
        hGh_mn = np.maximum(hGh_mn, 0)
        denom_mn = np.sqrt(hGh_mn)
        denom_mn = flooring_fn(denom_mn)
        P = h_mn / denom_mn
        P = P.transpose(1, 0, 2)
        Y_main = P.conj() @ Y_main
        Y_main = Y_main.transpose(1, 0, 2)

        # Concat
        Y_m, Y_n = np.split(Y_main, [1], axis=0)
        Y1, Y2, Y3 = np.split(Y_sub, [m, n - 1], axis=0)

        if ascend:
            Y = np.concatenate([Y1, Y_m, Y2, Y_n, Y3], axis=0)
        else:
            Y = np.concatenate([Y1, Y_n, Y2, Y_m, Y3], axis=0)

    return Y


def update_by_ip2_one_pair(
    separated_pair: np.ndarray,
    demix_filter_pair: np.ndarray,
    weight_pair: np.ndarray,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
) -> np.ndarray:
    r"""Update demixing filters by pairwise iterative projection.

    Args:
        separated_pair (numpy.ndarray):
            Separated spectrograms. \
            The shape is (2, n_bins, n_frames).
        demix_filter_pair (numpy.ndarray):
            Demixing filters to be updated. \
            The shape is (n_bins, 2, n_channels).
        weight_pair (numpy.ndarray):
            Weights of covariance at each frame. \
            (2, n_bins, n_frames)
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used. \
            Default: ``functools.partial(max_flooring, eps=1e-10)``.

    Returns:
        numpy.ndarray:
            Updated demixing filter pair. \
            The shape is (n_bins, 2, n_channels).
    """
    if flooring_fn is None:
        flooring_fn = identity

    W = demix_filter_pair
    Y = separated_pair
    varphi = weight_pair

    YY_Hermite = Y[:, np.newaxis, :, :] * Y[np.newaxis, :, :, :].conj()
    YY_Hermite = YY_Hermite.transpose(2, 0, 1, 3)  # (n_bins, 2, 2, n_frames)

    G_YY = varphi[:, :, np.newaxis, np.newaxis, :] * YY_Hermite
    G = np.mean(G_YY, axis=-1)  # (2, n_bins, 2, 2)

    G_m, G_n = G
    _, H = eigh2(G_m, G_n)  # (n_bins, 2, 2)
    h = H.transpose(2, 0, 1)  # (2, n_bins, 2)
    hGh = h[:, :, np.newaxis, :].conj() @ G @ h[:, :, :, np.newaxis]
    hGh = np.squeeze(hGh, axis=-1)  # (2, n_bins, 1)
    hGh = np.real(hGh)
    hGh = np.maximum(hGh, 0)
    denom = np.sqrt(hGh)
    denom = flooring_fn(denom)
    h = h / denom
    H = h.transpose(1, 2, 0)
    W_conj = W.transpose(0, 2, 1).conj() @ H

    W = W_conj.transpose(0, 2, 1).conj()

    return W
