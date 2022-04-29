from typing import Optional, Union, List, Callable

import numpy as np

__all__ = ["GradICA", "NaturalGradICA"]


class GradICAbase:
    """Base class of independent component analysis (ICA) using gradient descent

    Args:
        step_size (``float``):
            Step size of gradient descent. Default: ``1e-1``.
        callbacks (``Optional[Union[Callable[[GradICAbase], None], \
            List[Callable[[GradICAbase], None]]]]``):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_record_loss (``bool``):
            Record loss at each iteration of gradient descent if ``should_record_loss=True``.
            Default: True.
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        callbacks: Optional[
            Union[Callable[["GradICAbase"], None], List[Callable[["GradICAbase"], None]]]
        ] = None,
        should_record_loss: bool = True,
    ) -> None:
        self.step_size = step_size

        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None

        self.input = None
        self.should_record_loss = should_record_loss

        if self.should_record_loss:
            self.loss = []
        else:
            self.loss = None

    def __call__(self, input: np.ndarray, n_iter: int = 100, **kwargs) -> np.ndarray:
        """Separate multichannel time-domain signal.

        Args:
            input (``:class:numpy.ndarray``):
                Mixture signal in time-domain. (n_channels, n_samples)
        Returns:
            ``:class:numpy.ndarray``:
                Separated signal in time-domain. (n_sources, n_samples)
        """
        self.input = input.copy()

        self._reset(**kwargs)

        if self.should_record_loss:
            loss = self.compute_negative_loglikelihood()
            self.loss.append(loss)

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self)

        for _ in range(n_iter):
            self.update_once()

            if self.should_record_loss:
                loss = self.compute_negative_loglikelihood()
                self.loss.append(loss)

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)

        self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "GradICA("
        s += ")"

        return s.format(**self.__dict__)

    def _reset(self, **kwargs) -> None:
        """Reset attributes following on given keyword arguments.

        Args:
            kwargs:
                Set arguments as attributes of ICA.
        """
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        X = self.input

        n_channels, n_samples = X.shape
        n_sources = n_channels  # n_channels == n_sources

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_samples = n_samples

        if not hasattr(self, "demix_filter"):
            W = np.eye(n_sources, n_channels, dtype=np.float64)
        else:
            # To avoid overwriting ``demix_filter`` given by keyword arguments.
            W = self.demix_filter.copy()

        self.demix_filter = W
        self.output = self.separate(X, demix_filter=W)

    def update_once(self) -> None:
        """Update demixing filters once.
        """
        raise NotImplementedError("Implement 'update_once' method.")

    def separate(self, input: np.ndarray, demix_filter: np.ndarray) -> np.ndarray:
        """Separate ``input`` using ``demixing_filter``.

        Args:
            input (``:class:numpy.ndarray``):
                Mixture signal in time-domain. (n_channels, n_samples)
            demix_filter (``:class:numpy.ndarray``):
                Demixing filters to separate signal. (n_sources, n_channels)

        Returns:
            ``:class:numpy.ndarray``:
                Separated signal in time-domain.
        """
        output = demix_filter @ input

        return output

    def contrast_fn(self, input: np.ndarray) -> np.ndarray:
        """Contrast function.

        Contrast function corresponds to -log(y).

        Args:
            input (``:class:numpy.ndarray``):
                Separated signal in time-domain. (n_channels, n_samples)

        Returns:
            ``:class:numpy.ndarray``:
                Result of computation of contrast function.
        """
        raise NotImplementedError("Implement 'contrast_fn' method.")

    def score_fn(self, input: np.ndarray) -> np.ndarray:
        """Score function.

        Score function corresponds to partial derivative of contrast function.

        Args:
            input (``:class:numpy.ndarray``):
                Separated signal in time-domain. (n_channels, n_samples)

        Returns:
            ``:class:numpy.ndarray``:
                Result of computation of score function.
        """
        raise NotImplementedError("Implement 'score_fn' method.")

    def compute_negative_loglikelihood(self) -> float:
        """Compute negative log-likelihood.

        Returns:
            ``float``:
                Computed negative log-likelihood.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)  # (n_channels, n_samples)
        logdet = np.log(np.abs(np.linalg.det(W)))
        G = self.contrast_fn(Y)
        loss = np.sum(np.mean(G, axis=1)) - logdet

        return loss


class GradICA(GradICAbase):
    """Independent component analysis (ICA) using gradient descent

    Args:
        step_size (``float``):
            Step size of gradient descent. Default: ``1e-1``.
        callbacks (``Optional[Union[Callable[[GradICA], None], \
            List[Callable[[GradICA], None]]]]``):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_record_loss (``bool``):
            Record loss at each iteration of gradient descent if ``should_record_loss=True``.
            Default: True.
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        callbacks: Optional[
            Union[Callable[["GradICA"], None], List[Callable[["GradICA"], None]]]
        ] = None,
        should_record_loss=True,
    ) -> None:
        super().__init__(
            step_size=step_size, callbacks=callbacks, should_record_loss=should_record_loss
        )

    def update_once(self) -> None:
        """Update demixing filters once using gradient descent.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        phi = self.score_fn(Y)
        phi_Y = np.mean(phi[:, np.newaxis, :] * Y[np.newaxis, :, :], axis=-1)
        W_inv = np.linalg.inv(W)
        W_inv_trans = W_inv.transpose(1, 0)
        eye = np.eye(self.n_sources)

        delta = (phi_Y - eye) @ W_inv_trans
        W = W - self.step_size * delta

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y

    def contrast_fn(self, input: np.ndarray) -> np.ndarray:
        """Contrast function.

        Contrast function corresponds to -log(y).

        Args:
            input (``:class:numpy.ndarray``):
                Separated signal in time-domain. (n_channels, n_samples)

        Returns:
            ``:class:numpy.ndarray``:
                Result of computation of contrast function.
        """
        return np.abs(input)

    def score_fn(self, input: np.ndarray) -> np.ndarray:
        """Score function.

        Score function corresponds to partial derivative of contrast function.

        Args:
            input (``:class:numpy.ndarray``):
                Separated signal in time-domain. (n_channels, n_samples)

        Returns:
            ``:class:numpy.ndarray``:
                Result of computation of score function.
        """
        return np.sign(input)


class NaturalGradICA(GradICAbase):
    """Independent component analysis (ICA) using natural gradient descent

    Args:
        step_size (``float``):
            Step size of gradient descent. Default: ``1e-1``.
        callbacks (``Optional[Union[Callable[[NaturalGradICA], None], \
            List[Callable[[NaturalGradICA], None]]]]``):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_record_loss (``bool``):
            Record loss at each iteration of gradient descent if ``should_record_loss=True``.
            Default: True.
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        callbacks: Optional[
            Union[Callable[["GradICA"], None], List[Callable[["GradICA"], None]]]
        ] = None,
        should_record_loss=True,
    ) -> None:
        super().__init__(
            step_size=step_size, callbacks=callbacks, should_record_loss=should_record_loss
        )

    def update_once(self) -> None:
        """Update demixing filters once using natural gradient descent.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        phi = self.score_fn(Y)
        phi_Y = np.mean(phi[:, np.newaxis, :] * Y[np.newaxis, :, :], axis=-1)
        eye = np.eye(self.n_sources)

        delta = (phi_Y - eye) @ W
        W = W - self.step_size * delta

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y

    def contrast_fn(self, input: np.ndarray) -> np.ndarray:
        """Contrast function.

        Contrast function corresponds to -log(y).

        Args:
            input (``:class:numpy.ndarray``):
                Separated signal in time-domain. (n_channels, n_samples)

        Returns:
            ``:class:numpy.ndarray``:
                Result of computation of contrast function.
        """
        return np.abs(input)

    def score_fn(self, input: np.ndarray) -> np.ndarray:
        """Score function.

        Score function corresponds to partial derivative of contrast function.

        Args:
            input (``:class:numpy.ndarray``):
                Separated signal in time-domain. (n_channels, n_samples)

        Returns:
            ``:class:numpy.ndarray``:
                Result of computation of score function.
        """
        return np.sign(input)
