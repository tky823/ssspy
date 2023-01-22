import functools
import itertools
from typing import Callable, Optional

import numpy as np

from ..special.flooring import identity, max_flooring

EPS = 1e-10


def correlation_based_permutation_solver(
    sequence: np.ndarray,
    *args,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
    overwrite: bool = True,
) -> np.ndarray:
    r"""Solve permutation of estimated spectrograms.

    Group channels at each frequency bin according to correlations
    between frequencies [#murata2001approach]_.

    Args:
        sequence (numpy.ndarray):
            Array-like sequence of shape (n_bins, n_sources, n_frames).
        args (tuple of numpy.ndarray, optional):
            Positional arguments each of which is ``numpy.ndarray``.
            The shapes of each item should be (n_bins, n_sources, \*).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``partial(max_flooring, eps=1e-10)``.
        overwrite (bool):
            Overwrite ``sequence`` and ``args`` if ``overwrite=True``.
            Default: ``True``.

    Returns:
        - If ``args`` is not given, ``numpy.ndarray`` of permutated separated spectrograms
            with shape of (n_sources, n_bins, n_frames) are returned.
        - If one positional argument is given, ``numpy.ndarray``s of permutated separated
            spectrograms and the permutated positional argument are returned.
        - If more than two positional arguments are given, ``numpy.ndarray``s of
            permutated separated spectrograms and the permutated positional arguments are returned.

        .. [#murata2001approach]
            N. Murata, S. Ikeda, and A. Ziehe,
            "An approach to blind source separation based on temporal structure of speech signals,"
            in *Neurocomputing*, vol. 41, no. 1, pp. 1-24, 2001.

    .. note::

        In this function, the shape of ``separated`` is expected ``(n_bins, n_sources, ...)``,
        which is different from other functions.
    """
    assert sequence.ndim == 3, "Dimension of sequence is expected to be 3."

    for pos_idx, arg in enumerate(args):
        if arg.shape[:2] != sequence.shape[:2]:
            raise ValueError("The shape of {}th argument is invalid.".format(pos_idx + 1))

    if overwrite:
        Y = sequence
        permutable = args
    else:
        Y = sequence.copy()

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
    global_iter: int = 1,
    local_iter: int = 1,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
    multi_centroids: bool = False,
    overwrite: bool = True,
) -> np.ndarray:
    r"""Align permutations between frequencies based on score value [#sawada2010underdetermined]_.

    Args:
        sequence (numpy.ndarray):
            Array-like sequence of shape (n_bins, n_sources, n_frames).
        args (tuple of numpy.ndarray, optional):
            Positional arguments each of which is ``numpy.ndarray``.
            The shapes of each item should be (n_bins, n_sources, \*).
        global_iter (int):
            Number of iterations in global optimization.
        local_iter (int):
            Number of iterations in local optimization.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``partial(max_flooring, eps=1e-10)``.
        multi_centroids (bool):
            If ``multi_centroids=True``, multiple centroids are used in global optimization.
            However, this is not supported now. Default: ``False``.
        overwrite (bool):
            Overwrite ``sequence`` and ``args`` if ``overwrite=True``.
            Default: ``True``.

    .. [#sawada2010underdetermined]
        H. Sawada, S. Araki, and S. Makino,
        "Underdetermined convolutive blind source separation \
        via frequency bin-wise clustering and permutation alignment,"
        in *IEEE Trans. ASLP*, vol. 19, no. 3, pp. 516-527, 2010.
    """
    assert sequence.ndim == 3, "Dimension of sequence is expected to be 3."
    assert not multi_centroids, "multi_centroids version is not supported."

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

    if flooring_fn is None:
        flooring_fn = identity
    else:
        flooring_fn = flooring_fn

    n_bins, n_sources = sequence.shape[:2]
    na = np.newaxis
    eye = np.eye(n_sources)
    permutations = np.array(list(itertools.permutations(range(n_sources))))

    sequence_mean = sequence.mean(axis=-1, keepdims=True)
    sequence_std = sequence.std(axis=-1, keepdims=True)
    sequence_normalized = (sequence - sequence_mean) / sequence_std

    for _ in range(global_iter):
        centroid = sequence_normalized.mean(axis=0)
        centroid_std = centroid.std(axis=-1, keepdims=True)
        scores = []

        for perm in permutations:
            num = np.mean(sequence_normalized[:, perm, na] * centroid[na, :], axis=-1)
            denom = flooring_fn(centroid_std)
            corr = num / denom
            score = np.sum(eye * corr - (1 - eye) * corr, axis=(1, 2))
            scores.append(score)

        scores = np.stack(scores, axis=1)
        perm_max = np.argmax(scores, axis=1)
        perm_max = permutations[perm_max]
        sequence_normalized = _parallel_sort(sequence_normalized, perm_max)
        sequence = _parallel_sort(sequence, perm_max)

        for idx in range(len(permutable)):
            permutable[idx][:] = _parallel_sort(permutable[idx], perm_max)

    # local optimization
    for _ in range(local_iter):
        for bin_idx in range(n_bins):
            min_idx = max(0, bin_idx - 3)
            max_idx = min(n_bins - 1, bin_idx + 3)
            covariant_indices = set(range(min_idx, bin_idx)) | set(range(bin_idx + 1, max_idx + 1))

            min_idx = max(0, bin_idx // 2 - 1)
            max_idx = min(n_bins - 1, bin_idx // 2 + 1)
            covariant_indices |= set(range(min_idx, max_idx + 1))

            min_idx = max(0, 2 * bin_idx - 1)
            max_idx = min(n_bins - 1, 2 * bin_idx + 1)
            covariant_indices |= set(range(min_idx, max_idx + 1))

            # deterministic
            covariant_indices = sorted(list(covariant_indices))
            covariant_sequence = sequence_normalized[covariant_indices]

            scores = []

            for perm in permutations:
                num = np.mean(
                    sequence_normalized[bin_idx, perm, na] * covariant_sequence[:, na],
                    axis=-1,
                )
                denom = flooring_fn(centroid_std)
                corr = num / denom
                score = np.sum(eye * corr - (1 - eye) * corr, axis=(1, 2))
                score = score.sum(axis=0)
                scores.append(score)

            scores = np.stack(scores, axis=0)
            perm_max = np.argmax(scores, axis=0)
            perm_max = permutations[perm_max]
            sequence_normalized[bin_idx] = sequence_normalized[bin_idx, perm_max]
            sequence[bin_idx] = sequence[bin_idx, perm_max]

            for idx in range(len(permutable)):
                permutable[idx][bin_idx] = permutable[idx][bin_idx, perm_max]

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
