import warnings
from typing import Iterable, Optional, Tuple

from ..utils.select_pair import combination_pair_selector as combination_pair_selector_base
from ..utils.select_pair import sequential_pair_selector as sequential_pair_selector_base


def sequential_pair_selector(
    n_sources: int, stop: Optional[int] = None, step: int = 1, sort: bool = False
) -> Iterable[Tuple[int, int]]:
    r"""Select pair in pairwise update.

    Args:
        n_sources (int):
            Number of sources.
        step (int):
            This parameter determines step size.
            For instance, if ``sequential_pair_selector(n_sources=6, step=2, sort=False)``,
            this function yields ``0, 1``, ``2, 3``, ``4, 5``, ``0, 1``, ``2, 3``, ``4, 5``.
            Default: ``1``.
        sort (bool):
            Sort pair to ensure :math:`m<n` if ``sort=True``.
            Default: ``False``.

    Yields:
        Pair (tuple) of indices.

    Examples:
        .. code-block:: python

            >>> for m, n in combination_pair_selector(4):
            ...     print(m, n)
            0 1
            1 2
            2 3
            3 0
    """
    warnings.warn("Use ssspy.utils.select_pair.sequential_pair_selector instead.", UserWarning)

    yield from sequential_pair_selector_base(n_sources, stop=stop, step=step, sort=sort)


def combination_pair_selector(n_sources: int, sort: bool = False) -> Iterable[Tuple[int, int]]:
    r"""Select pair in pairwise update.

    Args:
        n_sources (int):
            Number of sources.
        sort (bool):
            Sort pair to ensure :math:`m<n` if ``sort=True``.
            Default: ``False``.

    Yields:
        Pair (tuple) of indices.

    Examples:
        .. code-block:: python

            >>> for m, n in combination_pair_selector(4):
            ...     print(m, n)
            0 1
            0 2
            0 3
            1 2
            1 3
            2 3
    """
    warnings.warn("Use ssspy.utils.select_pair.combination_pair_selector instead.", UserWarning)

    yield from combination_pair_selector_base(n_sources, sort=sort)
