import functools
import itertools
from typing import Callable, Optional

import numpy as np

from ..special.flooring import identity, max_flooring

EPS = 1e-10


def correlation_based_permutation_solver(
    separated: np.ndarray,
    *args,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
    overwrite: bool = True,
) -> np.ndarray:
    r"""Solve permutation of estimated spectrograms.

    Group channels at each frequency bin according to correlations
    between frequencies [#sawada2010underdetermined]_.

    Args:
        separated (numpy.ndarray):
            Separated spectrograms with shape of (n_sources, n_bins, n_frames).
        args (tuple of numpy.ndarray, optional):
            Positional arguments each of which is ``numpy.ndarray``.
            The shapes of each item should be (n_bins, n_sources, \*).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``partial(max_flooring, eps=1e-15)``.
        overwrite (bool):
            Overwrite ``demix_filter`` if ``overwrite=True``.
            Default: ``True``.

    Returns:
        - If ``args`` is not given, ``numpy.ndarray`` of permutated separated spectrograms
            with shape of (n_sources, n_bins, n_frames) are returned.
        - If one positional argument is given, ``numpy.ndarray``s of permutated separated
            spectrograms and the permutated positional argument are returned.
        - If more than two positional arguments are given, ``numpy.ndarray``s of
            permutated separated spectrograms and the permutated positional arguments are returned.

        .. [#sawada2010underdetermined]
            H. Sawada, S. Araki, and S. Makino,
            "Underdetermined convolutive blind source separation \
            via frequency bin-wise clustering and permutation alignment,"
            in *IEEE Trans. ASLP*, vol. 19, no. 3, pp. 516-527, 2010.
    """
    if overwrite:
        Y = separated
        sortable = args
    else:
        Y = separated.copy()

        sortable = []

        for arg in args:
            sortable.append(arg.copy())

        sortable = tuple(sortable)

    if flooring_fn is None:
        flooring_fn = identity
    else:
        flooring_fn = flooring_fn

    n_sources, n_bins, _ = Y.shape

    permutations = list(itertools.permutations(range(n_sources)))

    P = np.abs(Y).transpose(1, 0, 2)
    norm = np.sqrt(np.sum(P**2, axis=1, keepdims=True))
    norm = flooring_fn(norm)
    P = P / norm
    correlation = np.sum(P @ P.transpose(0, 2, 1), axis=(1, 2))
    indices = np.argsort(correlation)

    min_idx = indices[0]
    P_criteria = P[min_idx]

    for bin_idx in range(1, n_bins):
        min_idx = indices[bin_idx]
        P_max = None
        perm_max = None

        for perm in permutations:
            P_perm = np.sum(P_criteria * P[min_idx, perm, :])

            if P_max is None or P_perm > P_max:
                P_max = P_perm
                perm_max = perm

        P_criteria = P_criteria + P[min_idx, perm_max, :]

        Y[:, min_idx, :] = Y[perm_max, min_idx, :]

        for idx in range(len(sortable)):
            sortable[idx][min_idx, :] = sortable[idx][min_idx, perm_max]

    if len(sortable) == 0:
        return Y
    elif len(sortable) == 1:
        return Y, sortable[0]
    else:
        return Y, sortable
