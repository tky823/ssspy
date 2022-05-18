from typing import Optional, Union, List, Callable
import functools

import numpy as np

from ._flooring import max_flooring
from ..algorithm import projection_back

__all__ = ["GradIVA", "NaturalGradIVA", "GradLaplaceIVA", "NaturalGradLaplaceIVA"]

EPS = 1e-12


class IVAbase:
    r"""Base class of independent vector analysis (IVA) [#kim2006independent]_.

    Args:
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-12)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_apply_projection_back (bool):
            If ``should_apply_projection_back=True``, the projection back is applied to \
            estimated spectrograms. Default: ``True``.
        should_record_loss (bool):
            Record the loss at each iteration of the update algorithm \
            if ``should_record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.

    .. [#kim2006independent]
        T. Kim, H. T. Attias, S.-Y. Lee, and T.-W. Lee,
        "Blind source separation exploiting higher-order frequency dependencies,"
        in *IEEE Trans. ASLP*, vol. 15, no. 1, pp. 70-79, 2007.
    """

    def __init__(
        self,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["IVAbase"], None], List[Callable[["IVAbase"], None]]]
        ] = None,
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        self.flooring_fn = flooring_fn

        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None

        self.input = None
        self.should_apply_projection_back = should_apply_projection_back

        if reference_id is None and should_apply_projection_back:
            raise ValueError("Specify 'reference_id' if should_apply_projection_back=True.")
        else:
            self.reference_id = reference_id

        self.should_record_loss = should_record_loss

        if self.should_record_loss:
            self.loss = []
        else:
            self.loss = None

    def __call__(self, input: np.ndarray, n_iter: int = 100, **kwargs) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                Mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                Number of iterations of demixing filter updates.
                Default: 100.

        Returns:
            numpy.ndarray:
                The separated signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        raise NotImplementedError("Implement '__call__' method.")

    def __repr__(self) -> str:
        s = "IVA("
        s += ", should_apply_projection_back={should_apply_projection_back}"
        s += ", should_record_loss={should_record_loss}"

        if self.should_apply_projection_back:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(self, **kwargs) -> None:
        r"""Reset attributes following on given keyword arguments.

        Args:
            kwargs:
                Set arguments as attributes of IVA.
        """
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        X = self.input

        n_channels, n_bins, n_frames = X.shape
        n_sources = n_channels  # n_channels == n_sources

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        if not hasattr(self, "demix_filter"):
            W = np.eye(n_sources, n_channels, dtype=np.complex128)
            W = np.tile(W, reps=(n_bins, 1, 1))
        else:
            # To avoid overwriting ``demix_filter`` given by keyword arguments.
            W = self.demix_filter.copy()

        self.demix_filter = W
        self.output = self.separate(X, demix_filter=W)

    def separate(self, input: np.ndarray, demix_filter: np.ndarray) -> np.ndarray:
        r"""Separate ``input`` using ``demixing_filter``.

        .. math::
            \boldsymbol{y}_{ij}
            = \boldsymbol{W}_{i}\boldsymbol{x}_{ij}

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            demix_filter (numpy.ndarray):
                The demixing filters to separate ``input``.
                The shape is (n_bins, n_sources, n_channels).

        Returns:
            numpy.ndarray:
                The separated signal in frequency-domain.
                The shape is (n_sources, n_bins, n_frames).
        """
        X, W = input, demix_filter
        Y = W @ X.transpose(1, 0, 2)
        output = Y.transpose(1, 0, 2)

        return output

    def update_once(self) -> None:
        r"""Update demixing filters once.
        """
        raise NotImplementedError("Implement 'update_once' method.")

    def compute_negative_loglikelihood(self) -> float:
        r"""Compute negative log-likelihood :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L} \
            &= \frac{1}{J}\sum_{j,n}G(\vec{\boldsymbol{y}}_{jn}) \
            - 2\sum_{i}\log|\det\boldsymbol{W}_{i}|, \\
            G(\vec{\boldsymbol{y}}_{jn}) \
            &= - \log p(\vec{\boldsymbol{y}}_{jn})

        Returns:
            float:
                Computed negative log-likelihood.
        """
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")

    def apply_projection_back(self) -> None:
        r"""Apply projection back technique to estimated spectrograms.
        """
        assert self.should_apply_projection_back, "Set self.should_apply_projection_back=True."

        X, W = self.input, self.demix_filter
        W_scaled = projection_back(W, reference_id=self.reference_id)
        Y_scaled = self.separate(X, demix_filter=W_scaled)

        self.output, self.demix_filter = Y_scaled, W_scaled


class GradIVAbase(IVAbase):
    r""" Base class of independent vector analysis (IVA) using gradient descent.

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        contrast_fn (callable):
            A contrast function corresponds to :math:`-\log p(\vec{\boldsymbol{y}}_{jn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_frames).
        score_fn (callable):
            A score function corresponds to the partial derivative of the contrast function.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-12)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        should_apply_projection_back (bool):
            If ``should_apply_projection_back=True``, the projection back is applied to \
            estimated spectrograms. Default: ``True``.
        should_record_loss (bool):
            Record the loss at each iteration of the update algorithm \
            if ``should_record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        score_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["GradIVAbase"], None], List[Callable[["GradIVAbase"], None]]]
        ] = None,
        is_holonomic: bool = False,
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            should_apply_projection_back=should_apply_projection_back,
            should_record_loss=should_record_loss,
            reference_id=reference_id,
        )
        self.step_size = step_size

        if contrast_fn is None:
            raise ValueError("Specify contrast function.")
        else:
            self.contrast_fn = contrast_fn

        if score_fn is None:
            raise ValueError("Specify score function.")
        else:
            self.score_fn = score_fn

        self.is_holonomic = is_holonomic

    def __call__(self, input: np.ndarray, n_iter: int = 100, **kwargs) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                The number of iterations of demixing filter updates.
                Default: 100.

        Returns:
            numpy.ndarray:
                The separated signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
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

        if self.should_apply_projection_back:
            self.apply_projection_back()

        self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "GradIVA("
        s += "step_size={step_size}"
        s += ", is_holonomic={is_holonomic}"
        s += ", should_apply_projection_back={should_apply_projection_back}"
        s += ", should_record_loss={should_record_loss}"

        if self.should_apply_projection_back:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def compute_negative_loglikelihood(self) -> float:
        r"""Compute negative log-likelihood :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L} \
            &= \frac{1}{J}\sum_{j,n}G(\vec{\boldsymbol{y}}_{jn}) \
            - 2\sum_{i}\log|\det\boldsymbol{W}_{i}|, \\
            G(\vec{\boldsymbol{y}}_{jn}) \
            &= - \log p(\vec{\boldsymbol{y}}_{jn})

        Returns:
            float:
                Computed negative log-likelihood.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)  # (n_sources, n_bins, n_frames)
        logdet = np.log(np.abs(np.linalg.det(W)))  # (n_bins,)
        G = self.contrast_fn(Y)  # (n_sources, n_frames)
        loss = np.sum(np.mean(G, axis=1), axis=0) - 2 * np.sum(logdet, axis=0)

        return loss


class GradIVA(GradIVAbase):
    r"""Independent vector analysis (IVA) using gradient descent.

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        contrast_fn (callable):
            A contrast function corresponds to :math:`-\log p(\vec{\boldsymbol{y}}_{jn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_frames).
        score_fn (callable):
            A score function corresponds to the partial derivative of the contrast function.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-12)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        should_apply_projection_back (bool):
            If ``should_apply_projection_back=True``, the projection back is applied to \
            estimated spectrograms. Default: ``True``.
        should_record_loss (bool):
            Record the loss at each iteration of the update algorithm \
            if ``should_record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.

    Examples:
        .. code-block:: python

            def contrast_fn(y):
                return 2 * np.linalg.norm(y, axis=1)

            def score_fn(y):
                norm = np.linalg.norm(y, axis=1, keepdims=True)
                return y / np.maximum(norm, 1e-12)

            n_channels, n_bins, n_frames = 2, 2049, 128
            spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
                + 1j * np.random.randn(n_channels, n_bins, n_frames)

            iva = GradIVA(contrast_fn=contrast_fn, score_fn=score_fn)
            spectrogram_est = iva(spectrogram_mix, n_iter=5000)
            print(spectrogram_mix.shape, spectrogram_est.shape)
            >>> (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        score_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["GradIVA"], None], List[Callable[["GradIVA"], None]]]
        ] = None,
        is_holonomic: bool = True,
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            is_holonomic=is_holonomic,
            should_apply_projection_back=should_apply_projection_back,
            should_record_loss=should_record_loss,
            reference_id=reference_id,
        )

    def update_once(self) -> None:
        r"""Update demixing filters once using the gradient descent.

        If ``is_holonomic=True``, demixing filters are updated as follows:

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\boldsymbol{W}_{i} - \eta\left(\frac{1}{J}\sum_{j} \
            \boldsymbol{\phi}_{i}(\vec{\boldsymbol{Y}}_{j})\boldsymbol{y}_{ij}^{\mathsf{H}} \
            -\boldsymbol{I}\right)\boldsymbol{W}_{i}^{-\mathsf{H}},

        where

        .. math::
            \boldsymbol{\phi}_{i}(\vec{\boldsymbol{Y}}_{j})
            &= \left(\phi_{i}(\vec{\boldsymbol{y}}_{j1}),\ldots,\
            \phi_{i}(\vec{\boldsymbol{y}}_{jn}),\ldots,\
            \phi_{i}(\vec{\boldsymbol{y}}_{jN}))\
            \right)^{\mathsf{T}}\in\mathbb{C}^{N}, \\
            \phi_{i}(\vec{\boldsymbol{y}}_{jn})
            &= \frac{\partial G(\vec{\boldsymbol{y}}_{jn})}{\partial y_{ijn}^{*}}, \\
            G(\vec{\boldsymbol{y}}_{jn})
            &= -\log p(\vec{\boldsymbol{y}}_{jn}).

        Otherwise (``is_holonomic=False``),

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\boldsymbol{W}_{i}
            - \eta\cdot\mathrm{offdiag}\left(\frac{1}{J}\sum_{j}
            \boldsymbol{\phi}_{i}(\vec{\boldsymbol{Y}}_{j})\boldsymbol{y}_{ij}^{\mathsf{H}}\right)
            \boldsymbol{W}_{i}^{-\mathsf{H}}.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        Phi = self.score_fn(Y)
        Y_conj = Y.conj()
        PhiY = np.mean(Phi[:, np.newaxis, :, :] * Y_conj[np.newaxis, :, :, :], axis=-1)
        PhiY = PhiY.transpose(2, 0, 1)  # (n_bins, n_sources, n_sources)
        W_inv = np.linalg.inv(W)
        W_inv_Hermite = W_inv.transpose(0, 2, 1).conj()
        eye = np.eye(self.n_sources)

        if self.is_holonomic:
            delta = (PhiY - eye) @ W_inv_Hermite
        else:
            delta = ((1 - eye) * PhiY) @ W_inv_Hermite

        W = W - self.step_size * delta

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y


class NaturalGradIVA(GradIVAbase):
    r"""Independent vector analysis (IVA) using natural gradient descent.

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        contrast_fn (callable):
            A contrast function corresponds to :math:`-\log p(\vec{\boldsymbol{y}}_{jn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_frames).
        score_fn (callable):
            A score function corresponds to the partial derivative of the contrast function.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-12)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        should_apply_projection_back (bool):
            If ``should_apply_projection_back=True``, the projection back is applied to \
            estimated spectrograms. Default: ``True``.
        should_record_loss (bool):
            Record the loss at each iteration of the update algorithm \
            if ``should_record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.

    Examples:
        .. code-block:: python

            def contrast_fn(y):
                return 2 * np.linalg.norm(y, axis=1)

            def score_fn(y):
                norm = np.linalg.norm(y, axis=1, keepdims=True)
                return y / np.maximum(norm, 1e-12)

            n_channels, n_bins, n_frames = 2, 2049, 128
            spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
                + 1j * np.random.randn(n_channels, n_bins, n_frames)

            iva = NaturalGradIVA(contrast_fn=contrast_fn, score_fn=score_fn)
            spectrogram_est = iva(spectrogram_mix, n_iter=500)
            print(spectrogram_mix.shape, spectrogram_est.shape)
            >>> (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        score_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["NaturalGradIVA"], None], List[Callable[["NaturalGradIVA"], None]]]
        ] = None,
        is_holonomic: bool = True,
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            is_holonomic=is_holonomic,
            should_apply_projection_back=should_apply_projection_back,
            should_record_loss=should_record_loss,
            reference_id=reference_id,
        )

    def update_once(self) -> None:
        r"""Update demixing filters once using the natural gradient descent.

        If ``is_holonomic=True``, demixing filters are updated as follows:

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\boldsymbol{W}_{i} - \eta\left(\frac{1}{J}\sum_{j} \
            \boldsymbol{\phi}_{i}(\vec{\boldsymbol{Y}}_{j})\boldsymbol{y}_{ij}^{\mathsf{H}} \
            -\boldsymbol{I}\right)\boldsymbol{W}_{i},

        where

        .. math::
            \boldsymbol{\phi}_{i}(\vec{\boldsymbol{Y}}_{j})
            &= \left(\phi_{i}(\vec{\boldsymbol{y}}_{j1}),\ldots,\
            \phi_{i}(\vec{\boldsymbol{y}}_{jn}),\ldots,\
            \phi_{i}(\vec{\boldsymbol{y}}_{jN}))\
            \right)^{\mathsf{T}}\in\mathbb{C}^{N}, \\
            \phi_{i}(\vec{\boldsymbol{y}}_{jn})
            &= \frac{\partial G(\vec{\boldsymbol{y}}_{jn})}{\partial y_{ijn}^{*}}, \\
            G(\vec{\boldsymbol{y}}_{jn})
            &= -\log p(\vec{\boldsymbol{y}}_{jn}).

        Otherwise (``is_holonomic=False``),

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\boldsymbol{W}_{i}
            - \eta\cdot\mathrm{offdiag}\left(\frac{1}{J}\sum_{j}
            \boldsymbol{\phi}_{i}(\vec{\boldsymbol{Y}}_{j})\boldsymbol{y}_{ij}^{\mathsf{H}}\right)
            \boldsymbol{W}_{i}.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        Phi = self.score_fn(Y)
        Y_conj = Y.conj()
        PhiY = np.mean(Phi[:, np.newaxis, :, :] * Y_conj[np.newaxis, :, :, :], axis=-1)
        PhiY = PhiY.transpose(2, 0, 1)  # (n_bins, n_sources, n_sources)
        eye = np.eye(self.n_sources)

        if self.is_holonomic:
            delta = (PhiY - eye) @ W
        else:
            delta = ((1 - eye) * PhiY) @ W

        W = W - self.step_size * delta

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y


class GradLaplaceIVA(GradIVA):
    r"""Independent vector analysis (IVA) using the gradient descent on a Laplacian distribution.

    We assume :math:`\vec{\boldsymbol{y}}_{jn}` follows a Laplacian distribution.

    .. math::
        p(\vec{\boldsymbol{y}}_{jn})\propto\exp(\|\vec{\boldsymbol{y}}_{jn}\|_{2})

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-12)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        should_apply_projection_back (bool):
            If ``should_apply_projection_back=True``, the projection back is applied to \
            estimated spectrograms. Default: ``True``.
        should_record_loss (bool):
            Record the loss at each iteration of the update algorithm \
            if ``should_record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.

    Examples:
        .. code-block:: python

            n_channels, n_bins, n_frames = 2, 2049, 128
            spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
                + 1j * np.random.randn(n_channels, n_bins, n_frames)

            iva = GradLaplaceIVA()
            spectrogram_est = iva(spectrogram_mix, n_iter=5000)
            print(spectrogram_mix.shape, spectrogram_est.shape)
            >>> (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["GradLaplaceIVA"], None], List[Callable[["GradLaplaceIVA"], None]]]
        ] = None,
        is_holonomic: bool = True,
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        def contrast_fn(y: np.ndarray) -> np.ndarray:
            r"""Contrast function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray:
                    The shape is (n_sources, n_frames).
            """
            return 2 * np.linalg.norm(y, axis=1)

        def score_fn(y: np.ndarray) -> np.ndarray:
            r"""Score function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray:
                    The shape is (n_sources, n_bins, n_frames).
            """
            norm = np.linalg.norm(y, axis=1, keepdims=True)
            norm = self.flooring_fn(norm)
            return y / norm

        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            is_holonomic=is_holonomic,
            should_apply_projection_back=should_apply_projection_back,
            should_record_loss=should_record_loss,
            reference_id=reference_id,
        )

    def update_once(self) -> None:
        r"""Update demixing filters once using the gradient descent.

        If ``is_holonomic=True``, demixing filters are updated as follows:

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\boldsymbol{W}_{i} - \eta\left(\frac{1}{J}\sum_{j} \
            \boldsymbol{\phi}_{i}(\vec{\boldsymbol{Y}}_{j})\boldsymbol{y}_{ij}^{\mathsf{H}} \
            -\boldsymbol{I}\right)\boldsymbol{W}_{i}^{-\mathsf{H}},

        where

        .. math::
            \boldsymbol{\phi}_{i}(\vec{\boldsymbol{Y}}_{j})
            &= \left(\phi_{i}(\vec{\boldsymbol{y}}_{j1}),\ldots,\
            \phi_{i}(\vec{\boldsymbol{y}}_{jn}),\ldots,\
            \phi_{i}(\vec{\boldsymbol{y}}_{jN}))\
            \right)^{\mathsf{T}}\in\mathbb{C}^{N}, \\
            \phi_{i}(\vec{\boldsymbol{y}}_{jn})
            &= \frac{y_{ijn}}{\|\vec{\boldsymbol{y}}_{jn}\|_{2}}.

        Otherwise (``is_holonomic=False``),

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\boldsymbol{W}_{i}
            - \eta\cdot\mathrm{offdiag}\left(\frac{1}{J}\sum_{j}
            \boldsymbol{\phi}_{i}(\vec{\boldsymbol{Y}}_{j})\boldsymbol{y}_{ij}^{\mathsf{H}}\right)
            \boldsymbol{W}_{i}^{-\mathsf{H}}.
        """
        return super().update_once()

    def compute_negative_loglikelihood(self) -> float:
        r"""Compute negative log-likelihood :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L} \
            = \frac{2}{J}\sum_{j,n}\|\vec{\boldsymbol{y}}_{jn}\|_{2} \
            - 2\sum_{i}\log|\det\boldsymbol{W}_{i}|.

        Returns:
            float:
                Computed negative log-likelihood.
        """
        return super().compute_negative_loglikelihood()


class NaturalGradLaplaceIVA(NaturalGradIVA):
    r"""Independent vector analysis (IVA) using the natural gradient descent \
    on a Laplacian distribution.

    We assume :math:`\vec{\boldsymbol{y}}_{jn}` follows a Laplacian distribution.

    .. math::
        p(\vec{\boldsymbol{y}}_{jn})\propto\exp(\|\vec{\boldsymbol{y}}_{jn}\|_{2})

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-12)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        should_apply_projection_back (bool):
            If ``should_apply_projection_back=True``, the projection back is applied to \
            estimated spectrograms. Default: ``True``.
        should_record_loss (bool):
            Record the loss at each iteration of the update algorithm \
            if ``should_record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.

    Examples:
        .. code-block:: python

            n_channels, n_bins, n_frames = 2, 2049, 128
            spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
                + 1j * np.random.randn(n_channels, n_bins, n_frames)

            iva = NaturalGradLaplaceIVA()
            spectrogram_est = iva(spectrogram_mix, n_iter=500)
            print(spectrogram_mix.shape, spectrogram_est.shape)
            >>> (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[
                Callable[["NaturalGradLaplaceIVA"], None],
                List[Callable[["NaturalGradLaplaceIVA"], None]],
            ]
        ] = None,
        is_holonomic: bool = True,
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        def contrast_fn(y: np.ndarray) -> np.ndarray:
            r"""Contrast function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray:
                    The shape is (n_sources, n_frames).
            """
            return 2 * np.linalg.norm(y, axis=1)

        def score_fn(y: np.ndarray) -> np.ndarray:
            r"""Score function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray:
                    The shape is (n_sources, n_bins, n_frames).
            """
            norm = np.linalg.norm(y, axis=1, keepdims=True)
            norm = self.flooring_fn(norm)
            return y / norm

        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            is_holonomic=is_holonomic,
            should_apply_projection_back=should_apply_projection_back,
            should_record_loss=should_record_loss,
            reference_id=reference_id,
        )

    def update_once(self) -> None:
        r"""Update demixing filters once using the natural gradient descent.

        If ``is_holonomic=True``, demixing filters are updated as follows:

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\boldsymbol{W}_{i} - \eta\left(\frac{1}{J}\sum_{j} \
            \boldsymbol{\phi}_{i}(\vec{\boldsymbol{Y}}_{j})\boldsymbol{y}_{ij}^{\mathsf{H}} \
            -\boldsymbol{I}\right)\boldsymbol{W}_{i},

        where

        .. math::
            \boldsymbol{\phi}_{i}(\vec{\boldsymbol{Y}}_{j})
            &= \left(\phi_{i}(\vec{\boldsymbol{y}}_{j1}),\ldots,\
            \phi_{i}(\vec{\boldsymbol{y}}_{jn}),\ldots,\
            \phi_{i}(\vec{\boldsymbol{y}}_{jN}))\
            \right)^{\mathsf{T}}\in\mathbb{C}^{N}, \\
            \phi_{i}(\vec{\boldsymbol{y}}_{jn})
            &= \frac{y_{ijn}}{\|\vec{\boldsymbol{y}}_{jn}\|_{2}}.

        Otherwise (``is_holonomic=False``),

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\boldsymbol{W}_{i}
            - \eta\cdot\mathrm{offdiag}\left(\frac{1}{J}\sum_{j}
            \boldsymbol{\phi}_{i}(\vec{\boldsymbol{Y}}_{j})\boldsymbol{y}_{ij}^{\mathsf{H}}\right)
            \boldsymbol{W}_{i}.
        """
        return super().update_once()

    def compute_negative_loglikelihood(self) -> float:
        r"""Compute negative log-likelihood :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L} \
            = \frac{2}{J}\sum_{j,n}\|\vec{\boldsymbol{y}}_{jn}\|_{2} \
            - 2\sum_{i}\log|\det\boldsymbol{W}_{i}|.

        Returns:
            float:
                Computed negative log-likelihood.
        """
        return super().compute_negative_loglikelihood()
