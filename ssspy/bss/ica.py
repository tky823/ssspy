from typing import Optional, Union, List, Callable

import numpy as np

__all__ = ["GradICA", "NaturalGradICA", "GradLaplaceICA", "NaturalGradLaplaceICA"]


class GradICAbase:
    """Base class of independent component analysis (ICA) using gradient descent.

    Args:
        step_size (float):
            Step size of gradient descent. Default: ``1e-1``.
        callbacks (Optional[Union[Callable[[GradICAbase], None], \
        List[Callable[[GradICAbase], None]]]]):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        contrast_fn (Callable[[numpy.ndarray], numpy.ndarray]):
            Contrast function corresponds to -log(y).
            This function is expected to recieve (n_channels, n_samples)
            and return (n_channels, n_samples).
        score_fn (Callable[[numpy.ndarray], numpy.ndarray]):
            Score function corresponds to partial derivative of contrast function.
            This function is expected to recieve (n_channels, n_samples)
            and return (n_channels, n_samples).
        should_record_loss (bool):
            Record loss at each iteration of gradient descent if ``should_record_loss=True``.
            Default: ``True``.
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        score_fn: Callable[[np.ndarray], np.ndarray] = None,
        callbacks: Optional[
            Union[Callable[["GradICAbase"], None], List[Callable[["GradICAbase"], None]]]
        ] = None,
        should_record_loss: bool = True,
    ) -> None:
        self.step_size = step_size

        if contrast_fn is None:
            raise ValueError("Specify contrast function.")
        else:
            self.contrast_fn = contrast_fn

        if score_fn is None:
            raise ValueError("Specify score function.")
        else:
            self.score_fn = score_fn

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
            input (numpy.ndarray):
                Mixture signal in time-domain. Shape is (n_channels, n_samples).
            n_iter (int):
                Number of iterations of demixing filter updates.
                Default: 100.
        Returns:
            numpy.ndarray:
                Separated signal in time-domain. Shape is (n_sources, n_samples).
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
        s += "step_size={step_size}"
        s += ", should_record_loss={should_record_loss}"
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
            input (numpy.ndarray):
                Mixture signal in time-domain. (n_channels, n_samples)
            demix_filter (numpy.ndarray):
                Demixing filters to separate signal. (n_sources, n_channels)

        Returns:
            numpy.ndarray:
                Separated signal in time-domain.
        """
        output = demix_filter @ input

        return output

    def compute_negative_loglikelihood(self) -> float:
        """Compute negative log-likelihood.

        Returns:
            float:
                Computed negative log-likelihood.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)  # (n_channels, n_samples)
        logdet = np.log(np.abs(np.linalg.det(W)))
        G = self.contrast_fn(Y)
        loss = np.sum(np.mean(G, axis=1)) - logdet

        return loss


class GradICA(GradICAbase):
    """Independent component analysis (ICA) using gradient descent.

    Args:
        step_size (float):
            Step size of gradient descent. Default: ``1e-1``.
        callbacks (Optional[Union[Callable[[GradICA], None], \
        List[Callable[[GradICA], None]]]]):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        contrast_fn (Callable[[numpy.ndarray], numpy.ndarray]):
            Contrast function corresponds to -log(y).
            This function is expected to recieve (n_channels, n_samples)
            and return (n_channels, n_samples).
        score_fn (Callable[[numpy.ndarray], numpy.ndarray]):
            Score function corresponds to partial derivative of contrast function.
            This function is expected to recieve (n_channels, n_samples)
            and return (n_channels, n_samples).
        should_record_loss (bool):
            Record loss at each iteration of gradient descent if ``should_record_loss=True``.
            Default: ``True``.
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        score_fn: Callable[[np.ndarray], np.ndarray] = None,
        callbacks: Optional[
            Union[Callable[["GradICA"], None], List[Callable[["GradICA"], None]]]
        ] = None,
        should_record_loss: bool = True,
    ) -> None:
        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            callbacks=callbacks,
            should_record_loss=should_record_loss,
        )

    def update_once(self) -> None:
        """Update demixing filters once using gradient descent.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        Phi = self.score_fn(Y)
        Phi_Y = np.mean(Phi[:, np.newaxis, :] * Y[np.newaxis, :, :], axis=-1)
        W_inv = np.linalg.inv(W)
        W_inv_trans = W_inv.transpose(1, 0)
        eye = np.eye(self.n_sources)

        delta = (Phi_Y - eye) @ W_inv_trans
        W = W - self.step_size * delta

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y


class NaturalGradICA(GradICAbase):
    """Independent component analysis (ICA) using natural gradient descent.

    Args:
        step_size (float):
            Step size of gradient descent. Default: ``1e-1``.
        callbacks (Optional[Union[Callable[[GradICA], None], \
        List[Callable[[GradICA], None]]]]):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        contrast_fn (Callable[[numpy.ndarray], numpy.ndarray]):
            Contrast function corresponds to -log(y).
            This function is expected to recieve (n_channels, n_samples) \
            and return (n_channels, n_samples).
        score_fn (Callable[[numpy.ndarray], numpy.ndarray]):
            Score function corresponds to partial derivative of contrast function.
            This function is expected to recieve (n_channels, n_samples) \
            and return (n_channels, n_samples).
        should_record_loss (bool):
            Record loss at each iteration of gradient descent if ``should_record_loss=True``.
            Default: ``True``.
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        score_fn: Callable[[np.ndarray], np.ndarray] = None,
        callbacks: Optional[
            Union[Callable[["GradICA"], None], List[Callable[["GradICA"], None]]]
        ] = None,
        should_record_loss: bool = True,
    ) -> None:
        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            callbacks=callbacks,
            should_record_loss=should_record_loss,
        )

    def __repr__(self) -> str:
        s = "NaturalGradICA("
        s += "step_size={step_size}"
        s += ", should_record_loss={should_record_loss}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        """Update demixing filters once using natural gradient descent.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        Phi = self.score_fn(Y)
        Phi_Y = np.mean(Phi[:, np.newaxis, :] * Y[np.newaxis, :, :], axis=-1)
        eye = np.eye(self.n_sources)

        delta = (Phi_Y - eye) @ W
        W = W - self.step_size * delta

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y


class GradLaplaceICA(GradICA):
    """Independent component analysis (ICA) using gradient descent on Laplacian distribution.

    Args:
        step_size (float):
            Step size of gradient descent. Default: ``1e-1``.
        callbacks (Optional[Union[Callable[[GradLaplaceICA], None], \
        List[Callable[[GradLaplaceICA], None]]]]):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_record_loss (bool):
            Record loss at each iteration of gradient descent if ``should_record_loss=True``.
            Default: ``True``.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        callbacks: Optional[
            Union[Callable[["GradLaplaceICA"], None], List[Callable[["GradLaplaceICA"], None]]]
        ] = None,
        should_record_loss=True,
    ) -> None:
        def contrast_fn(input):
            return np.abs(input)

        def score_fn(input):
            return np.sign(input)

        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            callbacks=callbacks,
            should_record_loss=should_record_loss,
        )

    def __repr__(self) -> str:
        s = "GradLaplaceICA("
        s += "step_size={step_size}"
        s += ", should_record_loss={should_record_loss}"
        s += ")"

        return s.format(**self.__dict__)


class NaturalGradLaplaceICA(NaturalGradICA):
    """Independent component analysis (ICA) using natural gradient descent on Laplacian distribution.

    Args:
        step_size (float):
            Step size of gradient descent. Default: ``1e-1``.
        callbacks (Optional[Union[Callable[[NaturalGradLaplaceICA], None], \
        List[Callable[[NaturalGradLaplaceICA], None]]]]):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_record_loss (bool):
            Record loss at each iteration of gradient descent if ``should_record_loss=True``.
            Default: ``True``.
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        callbacks: Optional[
            Union[
                Callable[["NaturalGradLaplaceICA"], None],
                List[Callable[["NaturalGradLaplaceICA"], None]],
            ]
        ] = None,
        should_record_loss=True,
    ) -> None:
        def contrast_fn(input):
            return np.abs(input)

        def score_fn(input):
            return np.sign(input)

        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            callbacks=callbacks,
            should_record_loss=should_record_loss,
        )

    def __repr__(self) -> str:
        s = "NaturalGradLaplaceICA("
        s += "step_size={step_size}"
        s += ", should_record_loss={should_record_loss}"
        s += ")"

        return s.format(**self.__dict__)
