from typing import Optional, Union, List, Callable
import functools
import itertools

import numpy as np

from ._flooring import max_flooring
from ..algorithm import projection_back

__all__ = [
    "GradIVA",
    "NaturalGradIVA",
    "AuxIVA",
    "GradLaplaceIVA",
    "NaturalGradLaplaceIVA",
    "AuxLaplaceIVA",
]

algorithms_spatial = ["IP", "IP1", "IP2", "ISS", "ISS1", "ISS2"]
EPS = 1e-10


class IVAbase:
    r"""Base class of independent vector analysis (IVA) [#kim2006independent]_.

    Args:
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
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
        if flooring_fn is None:
            self.flooring_fn = lambda x: x
        else:
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
        s += "should_apply_projection_back={should_apply_projection_back}"
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
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)  # (n_sources, n_bins, n_frames)
        logdet = self.compute_logdet(W)  # (n_bins,)
        G = self.contrast_fn(Y)  # (n_sources, n_frames)
        loss = np.sum(np.mean(G, axis=1), axis=0) - 2 * np.sum(logdet, axis=0)

        return loss

    def compute_logdet(self, demix_filter: np.ndarray) -> np.ndarray:
        r"""Compute log-determinant of demixing filter
        """
        return np.log(np.abs(np.linalg.det(demix_filter)))  # (n_bins,)

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
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
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


class AuxIVAbase(IVAbase):
    r"""Base class of auxiliary-function-based independent vector analysis (IVA).

    Args:
        contrast_fn (callable):
            A contrast function corresponds to :math:`-\log p(\vec{\boldsymbol{y}}_{jn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_frames).
        d_contrast_fn (callable):
            A derivative of the contrast function.
            This function is expected to receive (n_channels, n_frames)
            and return (n_channels, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
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
    """

    def __init__(
        self,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        d_contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["AuxIVAbase"], None], List[Callable[["AuxIVAbase"], None]]]
        ] = None,
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
        self.contrast_fn = contrast_fn
        self.d_contrast_fn = d_contrast_fn

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
        return super().__call__(input, n_iter=n_iter, **kwargs)

    def __repr__(self) -> str:
        s = "AuxIVA("
        s += "should_apply_projection_back={should_apply_projection_back}"
        s += ", should_record_loss={should_record_loss}"

        if self.should_apply_projection_back:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)


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
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
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
                return y / np.maximum(norm, 1e-10)

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
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
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
                return y / np.maximum(norm, 1e-10)

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


class AuxIVA(AuxIVAbase):
    r"""Auxiliary-function-based independent vector analysis (IVA).

    Args:
        algorithm_spatial (str):
            Algorithm for demixing filter updates.
            Choose from "IP", "IP1", "IP2", "ISS", "ISS1", or "ISS2".
            Default: "IP".
        contrast_fn (callable):
            A contrast function corresponds to :math:`-\log p(\vec{\boldsymbol{y}}_{jn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_frames).
        d_contrast_fn (callable):
            A derivative of the contrast function.
            This function is expected to receive (n_channels, n_frames)
            and return (n_channels, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
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

    Examples:
        .. code-block:: python

            def contrast_fn(y):
                return 2 * np.linalg.norm(y, axis=1)

            def d_contrast_fn(y):
                return 2 * np.ones_like(y)

            n_channels, n_bins, n_frames = 2, 2049, 128
            spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
                + 1j * np.random.randn(n_channels, n_bins, n_frames)

            iva = AuxIVA(contrast_fn=contrast_fn, d_contrast_fn=d_contrast_fn)
            spectrogram_est = iva(spectrogram_mix, n_iter=100)
            print(spectrogram_mix.shape, spectrogram_est.shape)
            >>> (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        algorithm_spatial: str = "IP",
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        d_contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["AuxIVA"], None], List[Callable[["AuxIVA"], None]]]
        ] = None,
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
    ):
        super().__init__(
            contrast_fn=contrast_fn,
            d_contrast_fn=d_contrast_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            should_apply_projection_back=should_apply_projection_back,
            should_record_loss=should_record_loss,
            reference_id=reference_id,
        )

        assert algorithm_spatial in algorithms_spatial, "Not support {}.".format(algorithms_spatial)

        self.algorithm_spatial = algorithm_spatial

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

        if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
            self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "AuxIVA("
        s += "algorithm_spatial={algorithm_spatial}"
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
        super()._reset(**kwargs)

        if self.algorithm_spatial in ["ISS", "ISS1", "ISS2"]:
            self.demix_filter = None

    def _eigh(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        r"""Generalized eigendecomposition.
        """
        import scipy.linalg as splinalg

        h = []

        for A_i, B_i in zip(A, B):
            _, h_i = splinalg.eigh(A_i, B_i)  # (n_bins, 2, 2)
            h.append(h_i)

        h = np.stack(h, axis=0)

        return h

    def update_once(self) -> None:
        r"""Update demixing filters once.

        If ``self.algorithm_spatial`` is ``"IP"`` or ``"IP1"``, ``update_once_ip1`` is called.
        If ``self.algorithm_spatial`` is ``"IP2"``, ``update_once_ip2`` is called.
        """
        if self.algorithm_spatial in ["IP", "IP1"]:
            self.update_once_ip1()
        elif self.algorithm_spatial in ["IP2"]:
            self.update_once_ip2()
        elif self.algorithm_spatial in ["ISS", "ISS1"]:
            self.update_once_iss1()
        elif self.algorithm_spatial in ["ISS2"]:
            self.update_once_iss2()
        else:
            raise NotImplementedError("Not support {}.".format(self.algorithm_spatial))

    def update_once_ip1(self) -> None:
        r"""Update demixing filters once using iterative projection.

        Demixing filters are updated sequentially for :math:`n=1,\ldots,N` as follows:

        .. math::
            \boldsymbol{w}_{in}
            &\leftarrow\left(\boldsymbol{W}_{in}^{\mathsf{H}}\boldsymbol{U}_{in}\right)^{-1} \
            \boldsymbol{e}_{n}, \\
            \boldsymbol{w}_{in}
            &\leftarrow\frac{\boldsymbol{w}_{in}}
            {\sqrt{\boldsymbol{w}_{in}^{\mathsf{H}}\boldsymbol{U}_{in}\boldsymbol{w}_{in}}}, \\

        where

        .. math::
            \boldsymbol{U}_{in}
            &= \frac{1}{J}\sum_{j}
            \frac{G'_{\mathbb{R}}(\|\vec{\boldsymbol{y}}_{jn}\|_{2})}{2\|\vec{\boldsymbol{y}}_{jn}\|_{2}}
            \boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}, \\
            G(\vec{\boldsymbol{y}}_{jn})
            &= -\log p(\vec{\boldsymbol{y}}_{jn}), \\
            G_{\mathbb{R}}(\|\vec{\boldsymbol{y}}_{jn}\|_{2})
            &= G(\vec{\boldsymbol{y}}_{jn}).
        """
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        XX_Hermite = X[:, np.newaxis, :, :] * X[np.newaxis, :, :, :].conj()
        XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)  # (n_bins, n_channels, n_channels, n_frames)
        norm = np.linalg.norm(Y, axis=1)
        denominator = self.flooring_fn(2 * norm)
        weight = self.d_contrast_fn(norm) / denominator  # (n_sources, n_frames)
        GXX = weight[:, np.newaxis, np.newaxis, :] * XX_Hermite[:, np.newaxis, :, :, :]
        U = np.mean(GXX, axis=-1)  # (n_bins, n_sources, n_channels, n_channels)

        E = np.eye(n_sources, n_channels)  # (n_sources, n_channels)
        E = np.tile(E, reps=(n_bins, 1, 1))  # (n_bins, n_sources, n_channels)

        for src_idx in range(n_sources):
            w_n_Hermite = W[:, src_idx, :]  # (n_bins, n_channels)
            U_n = U[:, src_idx, :, :]
            e_n = E[:, src_idx, :]  # (n_bins, n_n_channels)

            WU = W @ U_n
            w_n = np.linalg.solve(WU, e_n)  # (n_bins, n_channels)
            wUw = w_n[:, np.newaxis, :].conj() @ U_n @ w_n[:, :, np.newaxis]
            wUw = np.real(wUw[..., 0])
            wUw = np.maximum(wUw, 0)
            denominator = np.sqrt(wUw)
            denominator = self.flooring_fn(denominator)
            w_n_Hermite = w_n.conj() / denominator
            W[:, src_idx, :] = w_n_Hermite

        self.demix_filter = W

    def update_once_ip2(self) -> None:
        r"""Update demixing filters once using pairwise iterative projection.

        For :math:`m` and :math:`n` (:math:`m\neq n`),
        compute weighted covariance matrix as follows:

        .. math::
            \boldsymbol{V}_{im}^{(m,n)}
            &= \frac{1}{J}\sum_{j}\frac{G'_{\mathbb{R}}(\|\vec{\boldsymbol{y}}_{jm}\|_{2})}
            {2\|\vec{\boldsymbol{y}}_{jm}\|_{2}} \
            \boldsymbol{y}_{ij}^{(m,n)}{\boldsymbol{y}_{ij}^{(m,n)}}^{\mathsf{H}} \\
            \boldsymbol{V}_{in}^{(m,n)}
            &= \frac{1}{J}\sum_{j}\frac{G'_{\mathbb{R}}(\|\vec{\boldsymbol{y}}_{jn}\|_{2})}
            {2\|\vec{\boldsymbol{y}}_{jn}\|_{2}} \
            \boldsymbol{y}_{ij}^{(m,n)}{\boldsymbol{y}_{ij}^{(m,n)}}^{\mathsf{H}},

        where

        .. math::
            \boldsymbol{y}_{ij}^{(m,n)}
            = \left(
            \begin{array}{c}
                \boldsymbol{w}_{im}^{\mathsf{H}}\boldsymbol{x}_{ij} \\
                \boldsymbol{w}_{in}^{\mathsf{H}}\boldsymbol{x}_{ij}
            \end{array}
            \right).

        Compute generalized eigenvectors of
        :math:`\boldsymbol{V}_{im}` and :math:`\boldsymbol{V}_{in}`.

        .. math::
            \boldsymbol{V}_{im}^{(m,n)}\boldsymbol{h}_{i}
            = \lambda_{i}\boldsymbol{V}_{in}^{(m,n)}\boldsymbol{h}_{i},

        where

        .. math::
            G(\vec{\boldsymbol{y}}_{jn})
            &= -\log p(\vec{\boldsymbol{y}}_{jn}), \\
            G_{\mathbb{R}}(\|\vec{\boldsymbol{y}}_{jn}\|_{2})
            &= G(\vec{\boldsymbol{y}}_{jn}).

        We denote two eigenvectors as :math:`\boldsymbol{h}_{im}`
        and :math:`\boldsymbol{h}_{in}`.

        .. math::
            \boldsymbol{h}_{im}
            &\leftarrow\frac{\boldsymbol{h}_{im}}
            {\sqrt{\boldsymbol{h}_{im}^{\mathsf{H}}\boldsymbol{V}_{im}^{(m,n)}
            \boldsymbol{h}_{im}}}, \\
            \boldsymbol{h}_{in}
            &\leftarrow\frac{\boldsymbol{h}_{in}}
            {\sqrt{\boldsymbol{h}_{in}^{\mathsf{H}}\boldsymbol{V}_{in}^{(m,n)}
            \boldsymbol{h}_{in}}}.

        Then, update :math:`\boldsymbol{w}_{im}` and :math:`\boldsymbol{w}_{in}`
        simultaneously.

        .. math::
            (
            \begin{array}{cc}
                \boldsymbol{w}_{im} & \boldsymbol{w}_{in}
            \end{array}
            )\leftarrow(
            \begin{array}{cc}
                \boldsymbol{w}_{im} & \boldsymbol{w}_{in}
            \end{array}
            )(
            \begin{array}{cc}
                \boldsymbol{h}_{im} & \boldsymbol{h}_{in}
            \end{array}
            )

        At each iteration, we update for all pairs of :math:`m`
        and :math:`n` (:math:`m<n`).
        """
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins

        X, W = self.input, self.demix_filter

        E = np.eye(n_sources, n_channels)  # (n_sources, n_channels)
        E = np.tile(E, reps=(n_bins, 1, 1))  # (n_bins, n_sources, n_channels)

        for m, n in itertools.combinations(range(n_sources), 2):
            W_mn = W[:, (m, n), :]  # (n_bins, 2, n_channels)
            Y_mn = self.separate(X, demix_filter=W_mn)  # (2, n_bins, n_frames)

            YY_Hermite = Y_mn[:, np.newaxis, :, :] * Y_mn[np.newaxis, :, :, :].conj()
            YY_Hermite = YY_Hermite.transpose(2, 0, 1, 3)  # (n_bins, 2, 2, n_frames)

            Y_mn_abs = np.linalg.norm(Y_mn, axis=1)  # (2, n_frames)
            denom_mn = self.flooring_fn(2 * Y_mn_abs)
            weight_mn = self.d_contrast_fn(Y_mn_abs) / denom_mn

            G_mn_YY = weight_mn[:, np.newaxis, np.newaxis, np.newaxis, :] * YY_Hermite
            U_mn = np.mean(G_mn_YY, axis=-1)  # (2, n_bins, 2, 2)

            U_m, U_n = U_mn
            H_mn = self._eigh(U_m, U_n)  # (n_bins, 2, 2)
            h_mn = H_mn.transpose(2, 0, 1)  # (2, n_bins, 2)
            hUh_mn = h_mn[:, :, np.newaxis, :].conj() @ U_mn @ h_mn[:, :, :, np.newaxis]
            hUh_mn = np.squeeze(hUh_mn, axis=-1)  # (2, n_bins, 1)
            hUh_mn = np.real(hUh_mn)
            hUh_mn = np.maximum(hUh_mn, 0)
            denom_mn = np.sqrt(hUh_mn)
            denom_mn = self.flooring_fn(denom_mn)
            h_mn = h_mn / denom_mn
            H_mn = h_mn.transpose(1, 2, 0)
            W_mn_conj = W_mn.transpose(0, 2, 1).conj() @ H_mn

            W[:, (m, n), :] = W_mn_conj.transpose(0, 2, 1).conj()

        self.demix_filter = W

    def update_once_iss1(self) -> None:
        n_sources = self.n_sources

        Y = self.output
        r = np.linalg.norm(Y, axis=1)
        denom = self.flooring_fn(2 * r)
        varphi = self.d_contrast_fn(r) / denom  # (n_sources, n_frames)

        for src_idx in range(n_sources):
            Y_n = Y[src_idx]  # (n_bins, n_frames)

            YY_n_conj = Y * Y_n.conj()
            YY_n = np.abs(Y_n) ** 2
            num = np.mean(varphi[:, np.newaxis, :] * YY_n_conj, axis=-1)
            denom = np.mean(varphi[:, np.newaxis, :] * YY_n, axis=-1)
            denom = self.flooring_fn(denom)
            v_n = num / denom
            v_n[src_idx] = 1 - 1 / np.sqrt(denom[src_idx])

            Y = Y - v_n[:, :, np.newaxis] * Y_n

        self.output = Y

    def update_once_iss2(self) -> None:
        r"""Update estimated spectrograms once using pairwise iterative source steering.

        Demixing filters are updated sequentially for :math:`n=1,\ldots,N` as follows:

        """
        raise NotImplementedError("Implement 'update_once_iss2' method.")

    def compute_negative_loglikelihood(self) -> float:
        if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
            return super().compute_negative_loglikelihood()
        else:
            X, Y = self.input, self.output
            X, Y = X.transpose(1, 0, 2), Y.transpose(1, 0, 2)
            X_Hermite = X.transpose(0, 2, 1).conj()
            XX_Hermite = X @ X_Hermite  # (n_bins, n_channels, n_channels)
            W = Y @ X_Hermite @ np.linalg.inv(XX_Hermite)
            logdet = self.compute_logdet(W)  # (n_bins,)
            G = self.contrast_fn(Y)  # (n_sources, n_frames)
            loss = np.sum(np.mean(G, axis=1), axis=0) - 2 * np.sum(logdet, axis=0)

            return loss

    def apply_projection_back(self) -> None:
        r"""Apply projection back technique to estimated spectrograms.
        """
        if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
            super().apply_projection_back()
        else:
            assert self.should_apply_projection_back, "Set self.should_apply_projection_back=True."

            X, Y = self.input, self.output
            Y_scaled = projection_back(Y, reference=X, reference_id=self.reference_id)

            self.output = Y_scaled


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
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
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
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
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


class AuxLaplaceIVA(AuxIVA):
    r"""Auxiliary-function-based independent vector analysis (IVA) \
    on a Laplacian distribution.

    We assume :math:`\vec{\boldsymbol{y}}_{jn}` follows a Laplacian distribution.

    .. math::
        p(\vec{\boldsymbol{y}}_{jn})\propto\exp(\|\vec{\boldsymbol{y}}_{jn}\|_{2})

    Args:
        algorithm_spatial (str):
            Algorithm for demixing filter updates.
            Choose from "IP", "IP1", or "IP2". Default: "IP".
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
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

    Examples:
        .. code-block:: python

            n_channels, n_bins, n_frames = 2, 2049, 128
            spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
                + 1j * np.random.randn(n_channels, n_bins, n_frames)

            iva = AuxLaplaceIVA()
            spectrogram_est = iva(spectrogram_mix, n_iter=100)
            print(spectrogram_mix.shape, spectrogram_est.shape)
            >>> (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        algorithm_spatial: str = "IP",
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["AuxLaplaceIVA"], None], List[Callable[["AuxLaplaceIVA"], None]]]
        ] = None,
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
    ):
        def contrast_fn(y):
            return 2 * np.linalg.norm(y, axis=1)

        def d_contrast_fn(y):
            return 2 * np.ones_like(y)

        super().__init__(
            algorithm_spatial=algorithm_spatial,
            contrast_fn=contrast_fn,
            d_contrast_fn=d_contrast_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            should_apply_projection_back=should_apply_projection_back,
            should_record_loss=should_record_loss,
            reference_id=reference_id,
        )
