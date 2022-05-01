from typing import Optional, Union, List, Callable

import numpy as np

__all__ = ["GradICA", "NaturalGradICA", "FastICA", "GradLaplaceICA", "NaturalGradLaplaceICA"]


class GradICAbase:
    r"""Base class of independent component analysis (ICA) using gradient descent.

    Args:
        step_size (float):
            Step size of gradient descent. Default: ``1e-1``.
        contrast_fn (callable):
            Contrast function corresponds to :math:`-\log(y_{nt})`.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        score_fn (callable):
            Score function corresponds to partial derivative of contrast function.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
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
        r"""Separate multichannel time-domain signal.

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
        r"""Reset attributes following on given keyword arguments.

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
        r"""Update demixing filters once.
        """
        raise NotImplementedError("Implement 'update_once' method.")

    def separate(self, input: np.ndarray, demix_filter: np.ndarray) -> np.ndarray:
        r"""Separate ``input`` using ``demixing_filter``.

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
        r"""Compute negative log-likelihood :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L} \
            &= \frac{1}{T}\sum_{t,n}G(y_{tn}) \
                - \log|\det\boldsymbol{W}| \\
            G(y_{tn}) \
            &= - \log p(y_{tn})

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


class FastICAbase:
    r"""Base class of fast independent component analysis (FastICA).

    Args:
        contrast_fn (callable):
            Contrast function corresponds to :math:`-\log(y_{nt})`.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        score_fn (callable):
            Score function corresponds to partial derivative of contrast function.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        d_score_fn (callable):
            Partial derivative of score function.
            This function is expected to return same shape tensor as the input.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_record_loss (bool):
            Record loss at each iteration of gradient descent if ``should_record_loss=True``.
            Default: ``True``.
    """

    def __init__(
        self,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        score_fn: Callable[[np.ndarray], np.ndarray] = None,
        d_score_fn: Callable[[np.ndarray], np.ndarray] = None,
        callbacks: Optional[
            Union[Callable[["FastICAbase"], None], List[Callable[["FastICAbase"], None]]]
        ] = None,
        should_record_loss: bool = True,
    ) -> None:
        if contrast_fn is None:
            raise ValueError("Specify contrast function.")
        else:
            self.contrast_fn = contrast_fn

        if score_fn is None:
            raise ValueError("Specify score function.")
        else:
            self.score_fn = score_fn

        if d_score_fn is None:
            raise ValueError("Specify derivative of score function.")
        else:
            self.d_score_fn = d_score_fn

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
        r"""Separate multichannel time-domain signal.

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

        self.output = self.separate(
            self.whitened_input, demix_filter=self.demix_filter, should_whiten=False
        )

        return self.output

    def __repr__(self) -> str:
        s = "FastICA("
        s += ", should_record_loss={should_record_loss}"
        s += ")"

        return s.format(**self.__dict__)

    def _reset(self, **kwargs) -> None:
        r"""Reset attributes following on given keyword arguments.

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

        XX_trans = np.mean(X[:, np.newaxis, :] * X[np.newaxis, :, :], axis=-1)
        lamb, Gamma = np.linalg.eigh(XX_trans)  # (n_channels,), (n_channels, n_channels)
        Lamb = np.diag(1 / np.sqrt(lamb))
        P = Lamb @ Gamma.transpose(1, 0)
        Z = P @ X

        self.proj_matrix = P
        self.whitened_input = Z
        self.demix_filter = W

        self.output = self.separate(Z, demix_filter=W)

    def update_once(self) -> None:
        r"""Update demixing filters once.
        """
        raise NotImplementedError("Implement 'update_once' method.")

    def separate(
        self, input: np.ndarray, demix_filter: np.ndarray, should_whiten=True
    ) -> np.ndarray:
        r"""Separate ``input`` using ``demixing_filter``.

        Args:
            input (numpy.ndarray):
                Mixture signal in time-domain. (n_channels, n_samples)
            demix_filter (numpy.ndarray):
                Demixing filters to separate signal. (n_sources, n_channels)
            should_whiten (bool):
                If ``should_whiten=True``, whitening (sphering) is applied to ``input``.
                Default: True.

        Returns:
            numpy.ndarray:
                Separated signal in time-domain.
        """
        if should_whiten:
            X = input
            XX_trans = np.mean(X[:, np.newaxis, :] * X[np.newaxis, :, :], axis=-1)
            lamb, Gamma = np.linalg.eigh(XX_trans)  # (n_channels,), (n_channels, n_channels)
            Lamb = np.diag(1 / np.sqrt(lamb))
            proj_matrix = Lamb @ Gamma.transpose(1, 0)
            whitened_input = proj_matrix @ X
        else:
            whitened_input = input

        output = demix_filter @ whitened_input

        return output

    def compute_negative_loglikelihood(self) -> float:
        r"""Compute negative log-likelihood :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L} \
            &= \frac{1}{T}\sum_{t,n}G(y_{tn}) \\
            G(y_{tn}) \
            &= - \log p(y_{tn})

        Returns:
            float:
                Computed negative log-likelihood.
        """
        Z, W = self.whitened_input, self.demix_filter
        Y = self.separate(Z, demix_filter=W, should_whiten=False)

        loss = np.mean(self.contrast_fn(Y), axis=-1)
        loss = loss.sum()

        return loss


class GradICA(GradICAbase):
    r"""Independent component analysis (ICA) using gradient descent.

    Args:
        step_size (float):
            Step size of gradient descent. Default: ``1e-1``.
        contrast_fn (callable):
            Contrast function corresponds to :math:`-\log(y_{nt})`.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        score_fn (callable):
            Score function corresponds to partial derivative of contrast function.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
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
        is_holonomic: bool = False,
        should_record_loss: bool = True,
    ) -> None:
        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            callbacks=callbacks,
            should_record_loss=should_record_loss,
        )

        self.is_holonomic = is_holonomic

    def __repr__(self) -> str:
        s = "GradICA("
        s += "step_size={step_size}"
        s += ", is_holonomic={is_holonomic}"
        s += ", should_record_loss={should_record_loss}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        r"""Update demixing filters once using gradient descent.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        Phi = self.score_fn(Y)
        PhiY = np.mean(Phi[:, np.newaxis, :] * Y[np.newaxis, :, :], axis=-1)
        W_inv = np.linalg.inv(W)
        W_inv_trans = W_inv.transpose(1, 0)
        eye = np.eye(self.n_sources)

        if self.is_holonomic:
            delta = (PhiY - eye) @ W_inv_trans
        else:
            delta = ((1 - eye) * PhiY) @ W_inv_trans

        W = W - self.step_size * delta

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y


class NaturalGradICA(GradICAbase):
    r"""Independent component analysis (ICA) using natural gradient descent.

    Args:
        step_size (float):
            Step size of gradient descent. Default: ``1e-1``.
        contrast_fn (callable):
            Contrast function corresponds to :math:`-\log(y_{nt})`.
            This function is expected to receive (n_channels, n_samples) \
            and return (n_channels, n_samples).
        score_fn (callable):
            Score function corresponds to partial derivative of contrast function.
            This function is expected to receive (n_channels, n_samples) \
            and return (n_channels, n_samples).
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
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
        is_holonomic: bool = False,
        should_record_loss: bool = True,
    ) -> None:
        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            callbacks=callbacks,
            should_record_loss=should_record_loss,
        )

        self.is_holonomic = is_holonomic

    def __repr__(self) -> str:
        s = "NaturalGradICA("
        s += "step_size={step_size}"
        s += ", is_holonomic={is_holonomic}"
        s += ", should_record_loss={should_record_loss}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        r"""Update demixing filters once using natural gradient descent.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        Phi = self.score_fn(Y)
        PhiY = np.mean(Phi[:, np.newaxis, :] * Y[np.newaxis, :, :], axis=-1)
        eye = np.eye(self.n_sources)

        if self.is_holonomic:
            delta = (PhiY - eye) @ W
        else:
            delta = ((1 - eye) * PhiY) @ W

        W = W - self.step_size * delta

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y


class FastICA(FastICAbase):
    r"""Fast independent component analysis (FastICA).

    Args:
        contrast_fn (callable):
            Contrast function corresponds to :math:`-\log(y_{nt})`.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        score_fn (callable):
            Score function corresponds to partial derivative of contrast function.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        d_score_fn (callable):
            Partial derivative of score function.
            This function is expected to return same shape tensor as the input.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_record_loss (bool):
            Record loss at each iteration of gradient descent if ``should_record_loss=True``.
            Default: ``True``.
    """

    def __init__(
        self,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        score_fn: Callable[[np.ndarray], np.ndarray] = None,
        d_score_fn: Callable[[np.ndarray], np.ndarray] = None,
        callbacks: Optional[
            Union[Callable[["FastICA"], None], List[Callable[["FastICA"], None]]]
        ] = None,
        should_record_loss: bool = True,
    ) -> None:
        super().__init__(
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            d_score_fn=d_score_fn,
            callbacks=callbacks,
            should_record_loss=should_record_loss,
        )

    def update_once(self) -> None:
        r"""Update demixing filters once using fixed-point iteration algorithm.
        """
        Z, W = self.whitened_input, self.demix_filter
        Y = self.separate(Z, demix_filter=W, should_whiten=False)

        for src_idx in range(self.n_sources):
            w_n = W[src_idx]  # (n_channels,)
            y_n = w_n @ Z  # (n_samples,)
            Gw_n = np.mean(self.d_score_fn(y_n), axis=-1) * w_n
            Gz = np.mean(self.score_fn(y_n) * Z, axis=-1)
            w_n = Gw_n - Gz

            if src_idx > 0:
                W_n = W[:src_idx]  # (src_idx - 1, n_channels)
                scale = np.sum(W_n * w_n, axis=-1, keepdims=True)
                w_n = w_n - np.sum(scale * W_n, axis=0)

            norm = np.linalg.norm(w_n)
            W[src_idx] = w_n / norm

        Y = self.separate(Z, demix_filter=W, should_whiten=False)

        self.demix_filter = W
        self.output = Y


class GradLaplaceICA(GradICA):
    r"""Independent component analysis (ICA) using gradient descent on Laplacian distribution.

    Args:
        step_size (float):
            Step size of gradient descent. Default: ``1e-1``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
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
        is_holonomic: bool = False,
        should_record_loss: bool = True,
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
            is_holonomic=is_holonomic,
            should_record_loss=should_record_loss,
        )

    def __repr__(self) -> str:
        s = "GradLaplaceICA("
        s += "step_size={step_size}"
        s += ", is_holonomic={is_holonomic}"
        s += ", should_record_loss={should_record_loss}"
        s += ")"

        return s.format(**self.__dict__)


class NaturalGradLaplaceICA(NaturalGradICA):
    r"""Independent component analysis (ICA) using natural gradient descent on Laplacian distribution.

    Args:
        step_size (float):
            Step size of gradient descent. Default: ``1e-1``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
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
        is_holonomic: bool = False,
        should_record_loss: bool = True,
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
            is_holonomic=is_holonomic,
            should_record_loss=should_record_loss,
        )

    def __repr__(self) -> str:
        s = "NaturalGradLaplaceICA("
        s += "step_size={step_size}"
        s += ", is_holonomic={is_holonomic}"
        s += ", should_record_loss={should_record_loss}"
        s += ")"

        return s.format(**self.__dict__)
