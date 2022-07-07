from typing import Optional, Union, List, Callable

import numpy as np

__all__ = ["GradICA", "NaturalGradICA", "FastICA", "GradLaplaceICA", "NaturalGradLaplaceICA"]


class GradICAbase:
    r"""Base class of independent component analysis (ICA) using the gradient descent.

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        contrast_fn (callable):
            A contrast function corresponds to :math:`-\log p(y_{tn})`.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        score_fn (callable):
            A score function corresponds to the partial derivative of the contrast function.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_record_loss (bool):
            Record the loss at each iteration of the gradient descent \
            if ``should_record_loss=True``.
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
        r"""Separate a time-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                Mixture signal in time-domain.
                The shape is (n_channels, n_samples).
            n_iter (int):
                Number of iterations of demixing filter updates.
                Default: 100.

        Returns:
            numpy.ndarray:
                The separated signal in time-domain.
                The shape is (n_sources, n_samples).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        if self.should_record_loss:
            loss = self.compute_loss()
            self.loss.append(loss)

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self)

        for _ in range(n_iter):
            self.update_once()

            if self.should_record_loss:
                loss = self.compute_loss()
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

        .. math::
            \boldsymbol{y}_{t}
            = \boldsymbol{W}\boldsymbol{x}_{t}

        Args:
            input (numpy.ndarray):
                The mixture signal in time-domain.
                The shape is (n_channels, n_samples).
            demix_filter (numpy.ndarray):
                The demixing filters to separate ``input``.
                The shape is (n_sources, n_channels).

        Returns:
            numpy.ndarray:
                The separated signal in time-domain.
                The shape is (n_sources, n_samples).
        """
        output = demix_filter @ input

        return output

    def compute_loss(self) -> float:
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
            A contrast function corresponds to :math:`-\log p(y_{tn})`.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        score_fn (callable):
            A score function corresponds to the partial derivative of the contrast function.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        d_score_fn (callable):
            A partial derivative of the score function.
            This function is expected to return the same shape tensor as the input.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_record_loss (bool):
            Record the loss at each of the fixed-point iteration \
            if ``should_record_loss=True``.
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
        r"""Separate a time-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                Mixture signal in time-domain.
                The shape is (n_channels, n_samples).
            n_iter (int):
                Number of iterations of demixing filter updates.
                Default: 100.

        Returns:
            numpy.ndarray:
                The separated signal in time-domain.
                The shape is (n_sources, n_samples).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        if self.should_record_loss:
            loss = self.compute_loss()
            self.loss.append(loss)

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self)

        for _ in range(n_iter):
            self.update_once()

            if self.should_record_loss:
                loss = self.compute_loss()
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

        If ``should_whiten=True``,

        .. math::
            \boldsymbol{y}_{t}
            &= \boldsymbol{W}\boldsymbol{z}_{t}, \\
            \boldsymbol{z}_{t}
            &= \boldsymbol{\Lambda}^{-\frac{1}{2}} \
            \boldsymbol{\Gamma}^{\mathsf{T}}\boldsymbol{x}_{t}, \\
            \boldsymbol{\Lambda}
            &:= \mathrm{diag}(\lambda_{1},\ldots,\lambda_{m},\ldots,\lambda_{M}) \
            \in\mathbb{R}^{M\times M}, \\
            \boldsymbol{\Gamma}
            &:= (\boldsymbol{\gamma}_{1}, \ldots,
            \boldsymbol{\gamma}_{m}, \ldots, \boldsymbol{\gamma}_{M}) \
            \in\mathbb{R}^{M\times M},

        where :math:`\lambda_{m}` and :math:`\boldsymbol{\gamma}_{m}` are
        an eigenvalue and eigenvector of
        :math:`\sum_{t}\boldsymbol{x}_{t}\boldsymbol{x}_{t}^{\mathsf{T}}`,
        respectively.

        Otherwise (``should_whiten=False``),

        .. math::
            \boldsymbol{y}_{t}
            = \boldsymbol{W}\boldsymbol{x}_{t}

        Args:
            input (numpy.ndarray):
                The mixture signal in time-domain.
                The shape is (n_channels, n_samples).
            demix_filter (numpy.ndarray):
                The demixing filters to separate ``input``.
                The shape is (n_sources, n_channels).
            should_whiten (bool):
                If ``should_whiten=True``, whitening (sphering) is applied to ``input``.
                Default: True.

        Returns:
            numpy.ndarray:
                The separated signal in time-domain.
                The shape is (n_sources, n_samples).
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

    def compute_loss(self) -> float:
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
    r"""Independent component analysis (ICA) using the gradient descent.

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        contrast_fn (callable):
            A contrast function corresponds to :math:`-\log p(y_{tn})`.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        score_fn (callable):
            A score function corresponds to the partial derivative of the contrast function.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        should_record_loss (bool):
            Record the loss at each iteration of the gradient descent \
            if ``should_record_loss=True``.
            Default: ``True``.

    Examples:
        .. code-block:: python

            def contrast_fn(y):
                return np.abs(y)

            def score_fn(y):
                return np.sign(y)

            n_channels, n_samples = 2, 160000
            waveform_mix = np.random.randn(n_channels, n_samples)

            ica = GradICA(contrast_fn=contrast_fn, score_fn=score_fn)
            waveform_est = ica(waveform_mix, n_iter=1000)
            print(waveform_mix.shape, waveform_est.shape)
            >>> (2, 160000), (2, 160000)
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
        r"""Update demixing filters once using the gradient descent.

        If ``is_holonomic=True``, demixing filters are updated as follows:

        .. math::
            \boldsymbol{W}
            \leftarrow\boldsymbol{W} - \eta\left(\frac{1}{T}\sum_{t} \
            \boldsymbol{\phi}(\boldsymbol{y}_{t})\boldsymbol{y}_{t}^{\mathsf{T}} \
            -\boldsymbol{I}\right)\boldsymbol{W}^{-\mathsf{T}},

        where

        .. math::
            \boldsymbol{\phi}(\boldsymbol{y}_{t})
            &= \left(\phi(y_{t1}),\ldots,\phi(y_{tN})\right)^{\mathsf{T}}\in\mathbb{R}^{N}, \\
            \phi(y_{tn})
            &= \frac{\partial G(y_{tn})}{\partial y_{tn}}, \\
            G(y_{tn})
            &= -\log p(y_{tn}).

        Otherwise (``is_holonomic=False``),

        .. math::
            \boldsymbol{W}
            \leftarrow\boldsymbol{W} - \eta\cdot\mathrm{offdiag}\left(\frac{1}{T}\sum_{t} \
            \boldsymbol{\phi}(\boldsymbol{y}_{t})\boldsymbol{y}_{t}^{\mathsf{T}}\right) \
            \boldsymbol{W}^{-\mathsf{T}}.
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
    r"""Independent component analysis (ICA) using the natural gradient descent [#amari1995new]_.

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        contrast_fn (callable):
            A contrast function corresponds to :math:`-\log p(y_{tn})`.
            This function is expected to receive (n_channels, n_samples) \
            and return (n_channels, n_samples).
        score_fn (callable):
            A score function corresponds to the partial derivative of the contrast function.
            This function is expected to receive (n_channels, n_samples) \
            and return (n_channels, n_samples).
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        should_record_loss (bool):
            Record the loss at each iteration of the gradient descent \
            if ``should_record_loss=True``.
            Default: ``True``.

    Examples:
        .. code-block:: python

            def contrast_fn(y):
                return np.abs(y)

            def score_fn(y):
                return np.sign(y)

            n_channels, n_samples = 2, 160000
            waveform_mix = np.random.randn(n_channels, n_samples)

            ica = NaturalGradICA(contrast_fn=contrast_fn, score_fn=score_fn)
            waveform_est = ica(waveform_mix, n_iter=100)
            print(waveform_mix.shape, waveform_est.shape)
            >>> (2, 160000), (2, 160000)

    .. [#amari1995new] S. Amari, A. Cichocki, and H. H. Yang,
        "A new learning algorithm for blind signal separation,"
        in *Proc. NIPS.*, pp. 757-763, 1996.
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
        r"""Update demixing filters once using the natural gradient descent.

        If ``is_holonomic=True``, demixing filters are updated as follows:

        .. math::
            \boldsymbol{W}
            \leftarrow\boldsymbol{W} - \eta\left(\frac{1}{T}\sum_{t} \
            \boldsymbol{\phi}(\boldsymbol{y}_{t})\boldsymbol{y}_{t}^{\mathsf{T}} \
            -\boldsymbol{I}\right)\boldsymbol{W},

        where

        .. math::
            \boldsymbol{\phi}(\boldsymbol{y}_{t})
            &= \left(\phi(y_{t1}),\ldots,\phi(y_{tN})\right)^{\mathsf{T}}\in\mathbb{R}^{N}, \\
            \phi(y_{tn})
            &= \frac{\partial G(y_{tn})}{\partial y_{tn}}, \\
            G(y_{tn})
            &= -\log p(y_{tn}).

        Otherwise (``is_holonomic=False``),

        .. math::
            \boldsymbol{W}
            \leftarrow\boldsymbol{W} - \eta\cdot\mathrm{offdiag}\left(\frac{1}{T}\sum_{t} \
            \boldsymbol{\phi}(\boldsymbol{y}_{t})\boldsymbol{y}_{t}^{\mathsf{T}}\right) \
            \boldsymbol{W}.
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
    r"""Fast independent component analysis (FastICA) [#hyvarinen1999fast]_.

    In FastICA, a whitening (sphering) is applied to input signal.

    .. math::
        \boldsymbol{z}_{t}
        &= \boldsymbol{\Lambda}^{-\frac{1}{2}} \
        \boldsymbol{\Gamma}^{\mathsf{T}}\boldsymbol{x}_{t}, \\
        \boldsymbol{\Lambda}
        &:= \mathrm{diag}(\lambda_{1},\ldots,\lambda_{m},\ldots,\lambda_{M}) \
        \in\mathbb{R}^{M\times M}, \\
        \boldsymbol{\Gamma}
        &:= (\boldsymbol{\gamma}_{1}, \ldots,
        \boldsymbol{\gamma}_{m}, \ldots, \boldsymbol{\gamma}_{M}) \
        \in\mathbb{R}^{M\times M},

    where :math:`\lambda_{m}` and :math:`\boldsymbol{\gamma}_{m}` are
    an eigenvalue and eigenvector of
    :math:`\sum_{t}\boldsymbol{x}_{t}\boldsymbol{x}_{t}^{\mathsf{T}}`,
    respectively.

    Furthermore, :math:`\boldsymbol{W}` is constrained to be orthogonal.

    .. math::
        \boldsymbol{W}\boldsymbol{W}^{\mathsf{T}}
        = \boldsymbol{I}

    Args:
        contrast_fn (callable):
            A contrast function corresponds to :math:`-\log p(y_{tn})`.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        score_fn (callable):
            A score function corresponds to the partial derivative of the contrast function.
            This function is expected to receive (n_channels, n_samples)
            and return (n_channels, n_samples).
        d_score_fn (callable):
            A Partial derivative of the score function.
            This function is expected to return the same shape tensor as the input.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_record_loss (bool):
            Record the loss at each of the fixed-point iteration \
            if ``should_record_loss=True``.
            Default: ``True``.

    Examples:
        .. code-block:: python

            def contrast_fn(y):
                return np.log(1 + np.exp(y))

            def score_fn(y):
                return 1 / (1 + np.exp(-y))

            def d_score_fn(y):
                sigmoid_y = 1 / (1 + np.exp(-y))
                return sigmoid_y * (1 - sigmoid_y)

            n_channels, n_samples = 2, 160000
            waveform_mix = np.random.randn(n_channels, n_samples)

            ica = FastICA(contrast_fn=contrast_fn, score_fn=score_fn, d_score_fn=d_score_fn)
            waveform_est = ica(waveform_mix, n_iter=10)
            print(waveform_mix.shape, waveform_est.shape)
            >>> (2, 160000), (2, 160000)

    .. [#hyvarinen1999fast] A. Hyvärinen,
        "Fast and Robust Fixed-Point Algorithms for Independent Component Analysis,"
        *IEEE Trans. on Neural Netw.*, vol. 10, no. 3, pp. 626-634, 1999.
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
        r"""Update demixing filters once using the fixed-point iteration algorithm.

        For :math:`n=1,\dots,N`, the demixing flter :math:`\boldsymbol{w}_{n}`
        is updated sequentially,

        .. math::
            y_{tn}
            &=\boldsymbol{w}_{n}^{\mathsf{T}}\boldsymbol{z}_{t}, \\
            \boldsymbol{w}_{n}^{+}
            &\leftarrow \frac{1}{T}\sum_{t}\phi(y_{tn})\boldsymbol{z}_{tn} \
            - \frac{1}{T}\sum_{t}\frac{\partial\phi(y_{tn})}{\partial y_{tn}} \
            \boldsymbol{w}_{n}, \\
            \boldsymbol{w}_{n}^{+}
            &\leftarrow\boldsymbol{w}_{n}^{+} \
            - \sum_{n'=1}^{n-1}\boldsymbol{w}_{n'}^{\mathsf{T}}\boldsymbol{w}_{n}^{+} \
            \boldsymbol{w}_{n}^{+}, \\
            \boldsymbol{w}_{n}
            &\leftarrow \frac{\boldsymbol{w}_{n}^{+}}{\|\boldsymbol{w}_{n}^{+}\|}.
        """
        Z, W = self.whitened_input, self.demix_filter

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
    r"""Independent component analysis (ICA) using the gradient descent on a Laplacian distribution.

    We assume :math:`y_{ijn}` follows a Laplacian distribution.

    .. math::
        p(y_{ijn})\propto\exp(|y_{ijn}|)

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        should_record_loss (bool):
            Record the loss at each iteration of the gradient descent \
            if ``should_record_loss=True``.
            Default: ``True``.

    Examples:
        .. code-block:: python

            n_channels, n_samples = 2, 160000
            waveform_mix = np.random.randn(n_channels, n_samples)

            ica = GradLaplaceICA()
            waveform_est = ica(waveform_mix, n_iter=1000)
            print(waveform_mix.shape, waveform_est.shape)
            >>> (2, 160000), (2, 160000)
    """

    def __init__(
        self,
        step_size: float = 1e-1,
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

    def update_once(self) -> None:
        r"""Update demixing filters once using the gradient descent.

        If ``is_holonomic=True``, demixing filters are updated as follows:

        .. math::
            \boldsymbol{W}
            \leftarrow\boldsymbol{W} - \eta\left(\frac{1}{T}\sum_{t} \
            \boldsymbol{\phi}(\boldsymbol{y}_{t})\boldsymbol{y}_{t}^{\mathsf{T}} \
            -\boldsymbol{I}\right)\boldsymbol{W}^{-\mathsf{T}},

        where

        .. math::
            \boldsymbol{\phi}(\boldsymbol{y}_{t})
            = \left(\mathrm{sign}(y_{t1}),\ldots,\mathrm{sign}(y_{tN})\right)^{\mathsf{T}} \
            \in\mathbb{R}^{N}.

        Otherwise (``is_holonomic=False``),

        .. math::
            \boldsymbol{W}
            \leftarrow\boldsymbol{W} - \eta\cdot\mathrm{offdiag}\left(\frac{1}{T}\sum_{t} \
            \boldsymbol{\phi}(\boldsymbol{y}_{t})\boldsymbol{y}_{t}^{\mathsf{T}}\right) \
            \boldsymbol{W}^{-\mathsf{T}}.
        """
        super().update_once()

    def compute_loss(self) -> float:
        r"""Compute negative log-likelihood :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L} \
            &= \frac{1}{T}\sum_{t,n}|y_{tn}| \
                - \log|\det\boldsymbol{W}| \\

        Returns:
            float:
                Computed negative log-likelihood.
        """
        return super().compute_loss()


class NaturalGradLaplaceICA(NaturalGradICA):
    r"""Independent component analysis (ICA) using the natural gradient descent \
    on a Laplacian distribution.

    We assume :math:`y_{ijn}` follows a Laplacian distribution.

    .. math::
        p(y_{ijn})\propto\exp(|y_{ijn}|)

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        should_record_loss (bool):
            Record the loss at each iteration of the gradient descent \
            if ``should_record_loss=True``.
            Default: ``True``.

    Examples:
        .. code-block:: python

            n_channels, n_samples = 2, 160000
            waveform_mix = np.random.randn(n_channels, n_samples)

            ica = NaturalGradLaplaceICA()
            waveform_est = ica(waveform_mix, n_iter=100)
            print(waveform_mix.shape, waveform_est.shape)
            >>> (2, 160000), (2, 160000)
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

    def update_once(self) -> None:
        r"""Update demixing filters once using the natural gradient descent.

        If ``is_holonomic=True``, demixing filters are updated as follows:

        .. math::
            \boldsymbol{W}
            \leftarrow\boldsymbol{W} - \eta\left(\frac{1}{T}\sum_{t} \
            \boldsymbol{\phi}(\boldsymbol{y}_{t})\boldsymbol{y}_{t}^{\mathsf{T}} \
            -\boldsymbol{I}\right)\boldsymbol{W},

        where

        .. math::
            \boldsymbol{\phi}(\boldsymbol{y}_{t})
            = \left(\mathrm{sign}(y_{t1}),\ldots,\mathrm{sign}(y_{tN})\right)^{\mathsf{T}} \
            \in\mathbb{R}^{N}.

        Otherwise (``is_holonomic=False``),

        .. math::
            \boldsymbol{W}
            \leftarrow\boldsymbol{W} - \eta\cdot\mathrm{offdiag}\left(\frac{1}{T}\sum_{t} \
            \boldsymbol{\phi}(\boldsymbol{y}_{t})\boldsymbol{y}_{t}^{\mathsf{T}}\right) \
            \boldsymbol{W}.
        """
        super().update_once()

    def compute_loss(self) -> float:
        r"""Compute negative log-likelihood :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L} \
            &= \frac{1}{T}\sum_{t,n}|y_{tn}| \
                - \log|\det\boldsymbol{W}| \\

        Returns:
            float:
                Computed negative log-likelihood.
        """
        return super().compute_loss()
