from typing import Iterable


def pair_selector(n_sources: int) -> Iterable[int, int]:
    r"""Select pair in pairwise update

    Args:
        n_sources: int

    Yields:
        int:
            First element of updating pair
        int:
            Seceond element of updating pair
    """
    for src_idx in range(n_sources):
        m, n = 2 * src_idx, 2 * src_idx + 1
        yield m % n_sources, n % n_sources
