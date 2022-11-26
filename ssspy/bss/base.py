from typing import Callable, List, Optional, Union

import numpy as np

__all__ = [
    "IterativeMethodBase",
]


class IterativeMethodBase:
    r"""Base class of iterative method.

    This class provides prototype of iterative updates.

    Args:
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
    """

    def __init__(
        self,
        callbacks: Optional[
            Union[
                Callable[["IterativeMethodBase"], None],
                List[Callable[["IterativeMethodBase"], None]],
            ]
        ] = None,
        record_loss: bool = True,
    ) -> None:
        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None

        self.record_loss = record_loss

        if self.record_loss:
            self.loss = []
        else:
            self.loss = None

    def __call__(self, *args, n_iter: int = 100, initial_call: bool = True, **kwargs) -> np.ndarray:
        r"""Iteratively call ``update_once``.

        Args:
            n_iter (int):
                The number of iterations of demixing filter updates.
                Default: ``100``.
            initial_call (bool):
                If ``True``, perform callbacks (and computation of loss if necessary)
                before iterations.
        """
        if initial_call:
            if self.record_loss:
                loss = self.compute_loss()
                self.loss.append(loss)

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)

        for _ in range(n_iter):
            self.update_once()

            if self.record_loss:
                loss = self.compute_loss()
                self.loss.append(loss)

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)

    def update_once(self) -> None:
        r"""Update parameters once."""
        raise NotImplementedError("Implement 'update_once' method.")

    def compute_loss(self) -> float:
        r"""Compute loss.

        Returns:
            Computed loss. The type is expected ``float``.
        """
        raise NotImplementedError("Implement 'compute_loss' method.")
