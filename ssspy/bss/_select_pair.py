from typing import Iterable, Tuple


def pair_selector(n_sources: int, sort: bool = True) -> Iterable[Tuple[int, int]]:
    r"""Select pair in pairwise update

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
    for src_idx in range(n_sources):
        m, n = 2 * src_idx, 2 * src_idx + 1
        m, n = m % n_sources, n % n_sources

        if sort:
            m, n = (n, m) if m > n else (m, n)

        yield m, n
