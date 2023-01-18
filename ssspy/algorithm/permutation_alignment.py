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
            Separated spectrograms with shape of (n_bins, n_sources, n_frames).
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

    .. note::

        In this function, the shape of ``separated`` is expected ``(n_bins, n_sources, ...)``,
        which is different from other functions.
    """
    for pos_idx, arg in enumerate(args):
        if arg.shape[:2] != separated.shape[:2]:
            raise ValueError("The shape of {}th argument is invalid.".format(pos_idx + 1))

    if overwrite:
        Y = separated
        permutable = args
    else:
        Y = separated.copy()

        permutable = []

        for arg in args:
            permutable.append(arg.copy())

        permutable = tuple(permutable)

    if flooring_fn is None:
        flooring_fn = identity
    else:
        flooring_fn = flooring_fn

    n_bins, n_sources, _ = Y.shape

    permutations = list(itertools.permutations(range(n_sources)))

    P = np.abs(Y)
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
        Y[min_idx, :] = Y[min_idx, perm_max]

        for idx in range(len(permutable)):
            permutable[idx][min_idx, :] = permutable[idx][min_idx, perm_max]

    if len(permutable) == 0:
        return Y
    elif len(permutable) == 1:
        return Y, permutable[0]
    else:
        return Y, permutable


def score_based_permutation_solver(
    sequence: np.ndarray,
    *args,
    n_iter: int = 1,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
    overwrite: bool = True,
) -> np.ndarray:
    """Align permutations between frequencies based on score value.

    Args:
        sequence (numpy.ndarray):
            Array-like sequence of shape (n_bins, n_sources, n_frames).
        args (tuple of numpy.ndarray, optional):
            Positional arguments each of which is ``numpy.ndarray``.
            The shapes of each item should be (n_bins, n_sources, \*).
        n_iter (int):
            Number of iterations in global optimization.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``partial(max_flooring, eps=1e-15)``.
        overwrite (bool):
            Overwrite ``sequence`` and ``args`` if ``overwrite=True``.
            Default: ``True``.
    """  # noqa: W605
    for pos_idx, arg in enumerate(args):
        if arg.shape[:2] != sequence.shape[:2]:
            raise ValueError("The shape of {}th argument is invalid.".format(pos_idx + 1))

    if overwrite:
        permutable = args
    else:
        sequence = sequence.copy()

        permutable = []

        for arg in args:
            permutable.append(arg.copy())

        permutable = tuple(permutable)

    n_sources = sequence.shape[1]
    na = np.newaxis
    eye = np.eye(n_sources)
    permutations = np.array(list(itertools.permutations(range(n_sources))))

    for _ in range(n_iter):
        centroid = sequence.mean(axis=0)

        sequence_normalized = sequence - sequence.mean(axis=-1, keepdims=True)
        centroid_normalized = centroid - centroid.mean(axis=-1, keepdims=True)

        sequence_std = sequence_normalized.std(axis=-1)
        centroid_std = centroid_normalized.std(axis=-1)

        scores = []

        for perm in permutations:
            num = np.mean(sequence_normalized[:, perm, na] * centroid_normalized[na, :], axis=-1)
            denom = sequence_std[:, perm, na] * centroid_std[na, perm]
            corr = num / flooring_fn(denom)

            score = np.sum(eye * corr - (1 - eye) * corr, axis=(1, 2))
            scores.append(score)

        scores = np.stack(scores, axis=1)
        perm_max = np.argmax(scores, axis=1)
        perm_max = permutations[perm_max]
        sequence = _parallel_sort(sequence, perm_max)

        for idx in range(len(permutable)):
            permutable[idx][:] = _parallel_sort(permutable[idx], perm_max)

    if len(permutable) == 0:
        return sequence
    elif len(permutable) == 1:
        return sequence, permutable[0]
    else:
        return sequence, permutable


def _parallel_sort(X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    shape = X.shape
    idx = np.repeat(indices, repeats=np.prod(shape[2:]), axis=-1).reshape(shape)
    X = np.take_along_axis(X, idx, axis=1)

    return X
