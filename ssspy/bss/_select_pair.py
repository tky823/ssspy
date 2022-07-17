from typing import Iterable, Tuple
import itertools


def sequential_pair_selector(n_sources: int, sort: bool = True) -> Iterable[Tuple[int, int]]:
    r"""Select pair in pairwise update.

    Args:
        n_sources (int):
            Number of sources.
        sort (bool):
            Sort pair to ensure :math:`m<n` if ``sort = True``.

    Yields:
        int:
            First element of updating pair.
        int:
            Second element of updating pair.
    """
    for m in range(n_sources):
        m, n = m % n_sources, (m + 1) % n_sources

        if sort:
            m, n = (n, m) if m > n else (m, n)

        yield m, n


def combination_pair_selector(n_sources: int, sort: bool = True) -> Iterable[Tuple[int, int]]:
    r"""Select pair in pairwise update.

    Args:
        n_sources (int):
            Number of sources.
        sort (bool):
            Sort pair to ensure :math:`m<n` if ``sort = True``.

    Yields:
        int:
            First element of updating pair.
        int:
            Second element of updating pair.
    """
    for m, n in itertools.combinations(range(n_sources), 2):

        if sort:
            m, n = (n, m) if m > n else (m, n)

        yield m, n
