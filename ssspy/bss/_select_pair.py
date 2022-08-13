from typing import Optional, Iterable, Tuple
import itertools


def sequential_pair_selector(
    n_sources: int, stop: Optional[int] = None, step: int = 1, sort: bool = False
) -> Iterable[Tuple[int, int]]:
    r"""Select pair in pairwise update.

    Args:
        n_sources (int):
            Number of sources.
        step (int):
            This parameter determines step size. \
            For instance, if `sequential_pair_selector(n_sources=6, step=2, sort=False)`, \
            this function yields `0, 1`, `2, 3`, `4, 5`, `0, 1`, `2, 3`, `4, 5`. \
            Default: `1`.
        sort (bool):
            Sort pair to ensure :math:`m<n` if ``sort=True``. \
            Default: ``False``.

    Yields:
        int:
            First element of updating pair.
        int:
            Second element of updating pair.
    """
    if stop is None:
        stop = n_sources

    for m in range(0, stop, step):
        m, n = m % n_sources, (m + 1) % n_sources

        if sort:
            m, n = (n, m) if m > n else (m, n)

        yield m, n


def combination_pair_selector(n_sources: int, sort: bool = False) -> Iterable[Tuple[int, int]]:
    r"""Select pair in pairwise update.

    Args:
        n_sources (int):
            Number of sources.
        sort (bool):
            Sort pair to ensure :math:`m<n` if ``sort=True``. \
            Default: ``False``.

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
