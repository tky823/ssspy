import functools
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from ..linalg import eigh2, inv2
from ..linalg.lqpqm import lqpqm2
from ..special.flooring import identity, max_flooring
from ..special.psd import to_psd
from ..utils.select_pair import sequential_pair_selector

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
            Demixing filters to be updated.
            The shape is (n_bins, n_sources, n_channels).
        weighted_covariance (numpy.ndarray):
            Weighted covariance matrix.
            The shape is (n_bins, n_sources, n_channels, n_channels).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        overwrite (bool):
            Overwrite ``demix_filter`` if ``overwrite=True``.
            Default: ``True``.

    Returns:
        numpy.ndarray of updated demixing filters.
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
    r"""Update demixing filters by pairwise iterative projection [#ono2018fast]_.

    Args:
        demix_filter (numpy.ndarray):
            Demixing filters to be updated.
            The shape is (n_bins, n_sources, n_channels).
        weighted_covariance (numpy.ndarray):
            Weighted covariance matrix.
            The shape is (n_bins, n_sources, n_channels, n_channels).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        pair_selector (callable, optional):
            Selector to choose updaing pair.
            If ``None`` is given, ``sequential_pair_selector`` is used.
            Default: ``None``.
        overwrite (bool):
            Overwrite ``demix_filter`` if ``overwrite=True``.
            Default: ``True``.

    Returns:
        numpy.ndarray of updated demixing filters.
        The shape is (n_bins, n_sources, n_channels).

    .. [#ono2018fast] N. Ono, \
        "Fast algorithm for independent component/vector/low-rank matrix analysis \
        with three or more sources," \
        in *Proc. ASJ Spring meeting*, 2018 (in Japanese).
    """
    if flooring_fn is None:
        flooring_fn = identity

    if pair_selector is None:
        pair_selector = sequential_pair_selector

    if overwrite:
        W = demix_filter
    else:
        W = demix_filter.copy()

    U = weighted_covariance

    _, n_sources, _ = W.shape

    for m, n in pair_selector(n_sources):
        pair = (m, n)
        W[:, pair, :] = update_by_ip2_one_pair(
            W, U[:, pair, :, :], pair=pair, flooring_fn=flooring_fn
        )

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
            Estimated spectrograms to be updated.
            The shape is (n_sources, n_bins, n_frames).
        weight (numpy.ndarray):
            Weights for estimated spectrogram.
            The shape is (n_sources, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.

    Returns:
        numpy.ndarray of updated spectrograms.
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
            Estimated spectrograms to be updated.
            The shape is (n_sources, n_bins, n_frames).
        weight (numpy.ndarray):
            Weights for estimated spectrogram.
            The shape is (n_sources, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        pair_selector (callable, optional):
            Selector to choose updaing pair.
            If ``None`` is given, ``sequential_pair_selector`` is used.
            Default: ``None``.

    Returns:
        numpy.ndarray of updated spectrograms.
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
    demix_filter: np.ndarray,
    weighted_covariance_pair: np.ndarray,
    pair: Tuple[int],
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
) -> np.ndarray:
    r"""Update demixing filters by pairwise iterative projection.

    Args:
        demix_filter (numpy.ndarray):
            Demixing filters.
            The shape is (n_bins, n_sources, n_channels).
        weighted_covariance_pair (numpy.ndarray):
            Weighted covariance matrix.
            The shape is (n_bins, 2, n_channels, n_channels).
        pair (tuple):
            Pair of source index to be updated.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.

    Returns:
        numpy.ndarray of updated demixing filter pair.
        The shape is (n_bins, 2, n_channels).
    """
    if flooring_fn is None:
        flooring_fn = identity

    m, n = pair
    W = demix_filter
    U_m, U_n = weighted_covariance_pair.transpose(1, 0, 2, 3)

    n_bins, n_sources, n_channels = W.shape

    E = np.eye(n_channels, n_sources)
    E_mn = E[:, (m, n)]
    E_mn = np.tile(E_mn, reps=(n_bins, 1, 1))

    WU_m = W @ U_m
    WU_n = W @ U_n

    P_m = np.linalg.solve(WU_m, E_mn)
    P_n = np.linalg.solve(WU_n, E_mn)

    PUP_m = P_m.transpose(0, 2, 1).conj() @ U_m @ P_m
    PUP_n = P_n.transpose(0, 2, 1).conj() @ U_n @ P_n

    _, H_mn = eigh2(PUP_m, PUP_n)
    H_mn = H_mn[..., ::-1]

    H_mn = H_mn.transpose(2, 0, 1)
    h_m, h_n = H_mn

    hUh_m = h_m[:, np.newaxis, :].conj() @ PUP_m @ h_m[:, :, np.newaxis]
    hUh_m = np.real(hUh_m[..., 0])
    hUh_m = np.maximum(hUh_m, 0)
    denom = np.sqrt(hUh_m)
    denom = flooring_fn(denom)
    h_m = h_m / denom

    hUh_n = h_n[:, np.newaxis, :].conj() @ PUP_n @ h_n[:, :, np.newaxis]
    hUh_n = np.real(hUh_n[..., 0])
    hUh_n = np.maximum(hUh_n, 0)
    denom = np.sqrt(hUh_n)
    denom = flooring_fn(denom)
    h_n = h_n / denom

    w_m = P_m @ h_m[..., np.newaxis]
    w_n = P_n @ h_n[..., np.newaxis]

    W_mn_conj = np.concatenate([w_m, w_n], axis=-1)
    W_mn = W_mn_conj.transpose(0, 2, 1).conj()

    return W_mn


def update_by_ipa(
    separated: np.ndarray,
    weight: np.ndarray,
    normalization: bool = True,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
    max_iter: int = 1,
) -> np.ndarray:
    r"""Update estimated spectrogram by iterative projection with adjustment (IPA).

    Args:
        separated (numpy.ndarray):
            Estimated spectrograms to be updated.
            The shape is (n_sources, n_bins, n_frames).
        weight (numpy.ndarray):
            Weights for estimated spectrogram.
            The shape is (n_sources, n_bins, n_frames).
        normalization (bool):
            If ``normalization=True``, normalization is applied to LQPQM.
            Default: ``True``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        max_iter (int):
            Maximum number of Newton-Raphson method. Default: ``1``.

    Returns:
        numpy.ndarray of estimated spectrograms of shape (n_sources, n_bins, n_frames).

    """
    if flooring_fn is None:
        flooring_fn = identity

    Y = separated
    varphi = weight

    n_sources = Y.shape[0]

    E = np.eye(n_sources)

    for source_idx in range(n_sources):
        YY_conj = Y[:, np.newaxis] * Y[np.newaxis, :].conj()
        U_tilde = np.mean(varphi[:, np.newaxis, np.newaxis] * YY_conj, axis=-1)
        U_tilde = U_tilde.transpose(3, 0, 1, 2)
        U_tilde = to_psd(U_tilde, axis1=-2, axis2=-1, flooring_fn=flooring_fn)

        E_n_left, e_n, E_n_right = np.split(E, [source_idx, source_idx + 1], axis=-1)
        E_n = np.concatenate([E_n_left, E_n_right], axis=-1)

        U_tilde_n = U_tilde[:, source_idx, :, :]
        U_tilde_n_inverse = _psd_inv(U_tilde_n, flooring_fn=flooring_fn)
        a_n = U_tilde[:, :, source_idx, source_idx]
        a_n = np.real(a_n)
        a_n = a_n @ E_n
        b_n = np.diagonal(U_tilde[:, :, source_idx, :], axis1=-2, axis2=-1)
        b_n = b_n @ E_n
        d_n = E_n.transpose(1, 0) @ U_tilde_n_inverse.conj()
        C_n = d_n @ E_n
        d_n = d_n[:, :, source_idx]

        Cd_n = np.linalg.solve(C_n, d_n)
        dCd_n = np.sum(d_n.conj() * Cd_n, axis=-1)
        dCd_n = np.real(dCd_n)
        eUe_n = U_tilde_n_inverse[:, source_idx, source_idx]
        eUe_n = np.real(eUe_n)
        z_n = eUe_n - dCd_n

        a_sqrt_n = np.sqrt(a_n)
        aa_n = a_sqrt_n[:, :, np.newaxis] * a_sqrt_n[:, np.newaxis, :]
        H_n = C_n / aa_n
        v_n = -b_n / a_sqrt_n - a_sqrt_n * Cd_n

        if normalization:
            trace = np.trace(H_n, axis1=-2, axis2=-1)
            trace = np.real(trace)

            H_n = H_n / trace[..., np.newaxis, np.newaxis]
            z_n = z_n / trace

        q_check_n = lqpqm2(
            H_n,
            v_n,
            z_n,
            flooring_fn=flooring_fn,
            singular_fn=lambda x: x < flooring_fn(0),
            max_iter=max_iter,
        )

        q_n = q_check_n / a_sqrt_n - b_n / a_n

        Eq_n = q_n.conj() @ E_n.transpose(1, 0)
        q_tilde_n = e_n.transpose(1, 0) - Eq_n

        Uq_n = np.linalg.solve(U_tilde_n, q_tilde_n)
        qUq_n = np.sum(q_tilde_n.conj() * Uq_n, axis=-1, keepdims=True)

        qUq_n = np.real(qUq_n)
        qUq_n = np.maximum(qUq_n, 0)
        denom = np.sqrt(qUq_n)
        denom = flooring_fn(denom)
        p_n = Uq_n / denom

        Y_n = Y[source_idx]
        p_n_conj = p_n.transpose(1, 0).conj()
        PY_n = np.sum(p_n_conj[..., np.newaxis] * Y, axis=0)
        PY_n = e_n[:, np.newaxis] * (PY_n - Y_n)
        Eq_n = Eq_n.transpose(1, 0)
        QY_n = Eq_n[:, :, np.newaxis] * Y_n

        Y = Y + PY_n + QY_n

    return Y


def update_by_block_decomposition_vcd(
    demix_filter: np.ndarray,
    weighted_covariance: np.ndarray,
    singular_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    overwrite: bool = True,
) -> np.ndarray:
    r"""
    Args:
        demix_filter (numpy.ndarray):
            Demixing filters to be updated.
            The shape is (n_blocks, n_neighbors, n_sources, n_channels).
        weighted_covariance (numpy.ndarray):
            Weighted covariance matrix.
            The shape is (n_blocks, n_neighbors, n_neighbors, n_sources, n_channels, n_channels).
        singular_fn (callable, optional):
            A flooring function to return singular condition.
            This function is expected to return the same shape bool tensor as the input.
            If ``singular_fn=None``,``lambda x: x == 0`` is used.
        overwrite (bool):
            Overwrite ``demix_filter`` if ``overwrite=True``.
            Default: ``True``.

    Returns:
        numpy.ndarray of updated demixing filters.
        The shape is (n_blocks, n_neighbors, n_sources, n_channels).
    """
    na = np.newaxis

    if singular_fn is None:

        def _is_zero(x: np.ndarray) -> np.ndarray:
            return x == 0

        singular_fn = _is_zero

    if overwrite:
        W = demix_filter
    else:
        W = demix_filter.copy()

    RXX = weighted_covariance
    U = np.diagonal(RXX, axis1=1, axis2=2)

    n_blocks, n_neighbors, n_sources, n_channels = W.shape

    E_i = np.eye(n_neighbors)
    E_n = np.eye(n_sources)
    E_n = np.tile(E_n, reps=(n_blocks, 1, 1))

    for neighbor_idx in range(n_neighbors):
        pad_mask_i = 1 - E_i[neighbor_idx]

        U_i = U[:, :, :, :, neighbor_idx]
        RXX_i = RXX[:, neighbor_idx]

        for source_idx in range(n_sources):
            e_n = E_n[:, source_idx, :]
            U_in = U_i[:, source_idx, :, :]
            RXX_in = RXX_i[:, :, source_idx]
            w_n_conj = W[:, :, source_idx, :].conj()

            RXY_in = RXX_in @ w_n_conj[:, :, :, na]

            gamma_in = np.sum(pad_mask_i[:, na] * RXY_in[..., 0], axis=1)

            WU_in = W[:, neighbor_idx, :, :] @ U_in
            eta_in = np.linalg.solve(WU_in, e_n)
            eta_hat_in = np.linalg.solve(U_in, gamma_in)
            eta_U_in = eta_in[:, na, :].conj() @ U_in

            xi_in = eta_U_in @ eta_in[:, :, na]
            xi_hat_in = eta_U_in @ eta_hat_in[:, :, na]

            xi_in = np.real(xi_in[..., 0])
            xi_in = np.maximum(xi_in, 0)
            xi_hat_in = xi_hat_in[..., 0]

            singular_condition = singular_fn(xi_hat_in)

            # to avoid zero division, but these will be ignored.
            xi_hat_in[singular_condition] = 1

            coeff = (xi_hat_in / (2 * xi_in)) * (
                1 - np.sqrt(1 + 4 * xi_in / (np.abs(xi_hat_in) ** 2))
            )
            coeff_singular = 1 / np.sqrt(xi_in)
            coeff = np.where(singular_condition, coeff_singular, coeff)

            w_in = coeff * eta_in - eta_hat_in

            W[:, neighbor_idx, source_idx, :] = w_in.conj()

    return W


def _psd_inv(
    X: np.ndarray,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
) -> np.ndarray:
    """Compute inversion of positive semidefinite matrix.

    Args:
        X (np.ndarray): Positive semidefinite matrix of shape (*, N, N).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.

    Returns:
        np.ndarray: Inversion of input matrix.

    """
    if flooring_fn is None:
        flooring_fn = identity

    Lamb, P = np.linalg.eigh(X)

    P_Hermite = P.swapaxes(-2, -1)

    if np.iscomplexobj(X):
        P_Hermite = P_Hermite.conj()

    Lamb_inv = 1 / flooring_fn(Lamb)
    Lamb_inv = Lamb_inv[..., np.newaxis] * np.eye(Lamb.shape[-1])

    return P @ Lamb_inv @ P_Hermite
