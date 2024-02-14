import functools
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

from ..algorithm import (
    MINIMAL_DISTORTION_PRINCIPLE_KEYWORDS,
    PROJECTION_BACK_KEYWORDS,
    minimal_distortion_principle,
    projection_back,
)
from ..linalg import eigh, prox
from ..special.flooring import identity, max_flooring
from ..transform import whiten
from ..utils.flooring import choose_flooring_fn
from ..utils.select_pair import sequential_pair_selector
from ._update_spatial_model import (
    update_by_ip1,
    update_by_ip2_one_pair,
    update_by_ipa,
    update_by_iss1,
    update_by_iss2,
)
from .admmbss import ADMMBSS
from .base import IterativeMethodBase
from .pdsbss import PDSBSS

__all__ = [
    "GradIVA",
    "NaturalGradIVA",
    "FastIVA",
    "FasterIVA",
    "AuxIVA",
    "PDSIVA",
    "ADMMIVA",
    "GradLaplaceIVA",
    "GradGaussIVA",
    "NaturalGradLaplaceIVA",
    "NaturalGradGaussIVA",
    "AuxLaplaceIVA",
    "AuxGaussIVA",
]

spatial_algorithms = ["IP", "IP1", "IP2", "ISS", "ISS1", "ISS2", "IPA"]
EPS = 1e-10


class IVABase(IterativeMethodBase):
    r"""Base class of independent vector analysis (IVA).

    Args:
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.
    """

    def __init__(
        self,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["IVABase"], None], List[Callable[["IVABase"], None]]]
        ] = None,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(callbacks=callbacks, record_loss=record_loss)

        if flooring_fn is None:
            self.flooring_fn = identity
        else:
            self.flooring_fn = flooring_fn

        self.input = None
        self.scale_restoration = scale_restoration

        if reference_id is None and scale_restoration:
            raise ValueError("Specify 'reference_id' if scale_restoration=True.")
        else:
            self.reference_id = reference_id

    def __call__(
        self, input: np.ndarray, n_iter: int = 100, initial_call: bool = True, **kwargs
    ) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                Mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                Number of iterations of demixing filter updates.
                Default: ``100``.
            initial_call (bool):
                If ``True``, perform callbacks (and computation of loss if necessary)
                before iterations.

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_channels, n_bins, n_frames).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        raise NotImplementedError("Implement '__call__' method.")

    def __repr__(self) -> str:
        s = "IVA("
        s += "scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(self, **kwargs) -> None:
        r"""Reset attributes by given keyword arguments.

        Args:
            kwargs:
                Keyword arguments to set as attributes of IVA.
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
            if self.demix_filter is None:
                W = None
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
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_sources, n_bins, n_frames).
        """
        X, W = input, demix_filter
        Y = W @ X.transpose(1, 0, 2)
        output = Y.transpose(1, 0, 2)

        return output

    def update_once(self) -> None:
        r"""Update demixing filters once."""
        raise NotImplementedError("Implement 'update_once' method.")

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L} \
            &= \frac{1}{J}\sum_{j,n}G(\vec{\boldsymbol{y}}_{jn}) \
            - 2\sum_{i}\log|\det\boldsymbol{W}_{i}|, \\
            G(\vec{\boldsymbol{y}}_{jn}) \
            &= - \log p(\vec{\boldsymbol{y}}_{jn})

        Returns:
            Computed loss.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)  # (n_sources, n_bins, n_frames)
        logdet = self.compute_logdet(W)  # (n_bins,)
        G = self.contrast_fn(Y)  # (n_sources, n_frames)
        loss = np.sum(np.mean(G, axis=1), axis=0) - 2 * np.sum(logdet, axis=0)
        loss = loss.item()

        return loss

    def compute_logdet(self, demix_filter: np.ndarray) -> np.ndarray:
        r"""Compute log-determinant of demixing filter.

        Args:
            demix_filter (numpy.ndarray):
                Demixing filters with shape of (n_bins, n_sources, n_channels).

        Returns:
            numpy.ndarray of computed log-determinant values.
        """
        _, logdet = np.linalg.slogdet(demix_filter)  # (n_bins,)

        return logdet

    def restore_scale(self) -> None:
        r"""Restore scale ambiguity.

        If ``self.scale_restoration=projection_back``, we use projection back technique.
        If ``self.scale_restoration=minimal_distortion_principle``,
        we use minimal distortion principle.
        """
        scale_restoration = self.scale_restoration

        assert scale_restoration, "Set self.scale_restoration=True."

        if type(scale_restoration) is bool:
            scale_restoration = PROJECTION_BACK_KEYWORDS[0]

        if scale_restoration in PROJECTION_BACK_KEYWORDS:
            self.apply_projection_back()
        elif scale_restoration in MINIMAL_DISTORTION_PRINCIPLE_KEYWORDS:
            self.apply_minimal_distortion_principle()
        else:
            raise ValueError("{} is not supported for scale restoration.".format(scale_restoration))

    def apply_projection_back(self) -> None:
        r"""Apply projection back technique to estimated spectrograms."""
        assert self.scale_restoration, "Set self.scale_restoration=True."

        X, W = self.input, self.demix_filter
        W_scaled = projection_back(W, reference_id=self.reference_id)
        Y_scaled = self.separate(X, demix_filter=W_scaled)

        self.output, self.demix_filter = Y_scaled, W_scaled

    def apply_minimal_distortion_principle(self) -> None:
        r"""Apply minimal distortion principle to estimated spectrograms."""
        assert self.scale_restoration, "Set self.scale_restoration=True."

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        Y_scaled = minimal_distortion_principle(Y, reference=X, reference_id=self.reference_id)
        X = X.transpose(1, 0, 2)
        Y = Y_scaled.transpose(1, 0, 2)
        X_Hermite = X.transpose(0, 2, 1).conj()
        W_scaled = Y @ X_Hermite @ np.linalg.inv(X @ X_Hermite)

        self.output, self.demix_filter = Y_scaled, W_scaled


class GradIVABase(IVABase):
    r"""Base class of independent vector analysis (IVA) using gradient descent.

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        contrast_fn (callable):
            A contrast function which corresponds to :math:`-\log p(\vec{\boldsymbol{y}}_{jn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_frames).
        score_fn (callable):
            A score function which corresponds to the partial derivative of the contrast function.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the gradient descent if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.
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
            Union[Callable[["GradIVABase"], None], List[Callable[["GradIVABase"], None]]]
        ] = None,
        is_holonomic: bool = False,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
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

    def __call__(
        self, input: np.ndarray, n_iter: int = 100, initial_call: bool = True, **kwargs
    ) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain. \
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                The number of iterations of demixing filter updates. \
                Default: ``100``.
            initial_call (bool):
                If ``True``, perform callbacks (and computation of loss if necessary)
                before iterations.

        Returns:
            numpy.ndarray:
                The separated signal in frequency-domain. \
                The shape is (n_channels, n_bins, n_frames).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        # Call __call__ of IVABase's parent, i.e. __call__ of IterativeMethodBase
        super(IVABase, self).__call__(n_iter=n_iter, initial_call=initial_call)

        if self.scale_restoration:
            self.restore_scale()

        self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "GradIVA("
        s += "step_size={step_size}"
        s += ", is_holonomic={is_holonomic}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)


class FastIVABase(IVABase):
    r"""Base class of fast independent vector analysis (FastIVA).

    Args:
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.
    """

    def __init__(
        self,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["IVABase"], None], List[Callable[["IVABase"], None]]]
        ] = None,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

    def __repr__(self) -> str:
        s = "FastIVA("
        s += "scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(self, **kwargs) -> None:
        super()._reset(**kwargs)

        X, W = self.input, self.demix_filter

        Z = whiten(X)

        Y = self.separate(Z, demix_filter=W, use_whitening=False)

        self.whitened_input = Z
        self.output = Y

    def separate(
        self, input: np.ndarray, demix_filter: np.ndarray, use_whitening: bool = True
    ) -> np.ndarray:
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
            use_whitening (bool):
                If ``use_whitening=True``, use_whitening (sphering) is applied to ``input``.
                Default: True.

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_sources, n_bins, n_frames).
        """
        if use_whitening:
            whitened_input = whiten(input)
        else:
            whitened_input = input

        output = super().separate(whitened_input, demix_filter=demix_filter)

        return output

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L} \
            &= \frac{1}{J}\sum_{j,n}G(\vec{\boldsymbol{y}}_{jn}), \\
            G(\vec{\boldsymbol{y}}_{jn}) \
            &= - \log p(\vec{\boldsymbol{y}}_{jn})

        Returns:
            Computed loss.
        """
        Z, W = self.whitened_input, self.demix_filter
        Y = self.separate(Z, demix_filter=W, use_whitening=False)  # (n_sources, n_bins, n_frames)

        G = self.contrast_fn(Y)  # (n_sources, n_frames)
        loss = np.sum(np.mean(G, axis=1), axis=0).item()

        return loss

    def apply_projection_back(self) -> None:
        r"""Apply projection back technique to estimated spectrograms."""
        assert self.scale_restoration, "Set self.scale_restoration=True."

        reference_id = self.reference_id

        X, Z = self.input, self.whitened_input
        W = self.demix_filter

        Y = self.separate(Z, demix_filter=W, use_whitening=False)
        Y_scaled = projection_back(Y, reference=X, reference_id=reference_id)

        Z = Z.transpose(1, 0, 2)
        Z_Hermite = Z.transpose(0, 2, 1).conj()
        ZZ_Hermite = Z @ Z_Hermite
        W_scaled = Y_scaled.transpose(1, 0, 2) @ Z_Hermite @ np.linalg.inv(ZZ_Hermite)

        self.output, self.demix_filter = Y_scaled, W_scaled


class AuxIVABase(IVABase):
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
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.
    """

    def __init__(
        self,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        d_contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["AuxIVABase"], None], List[Callable[["AuxIVABase"], None]]]
        ] = None,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )
        self.contrast_fn = contrast_fn
        self.d_contrast_fn = d_contrast_fn

    def __call__(
        self, input: np.ndarray, n_iter: int = 100, initial_call: bool = True, **kwargs
    ) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                The number of iterations of demixing filter updates.
                Default: ``100``.
            initial_call (bool):
                If ``True``, perform callbacks (and computation of loss if necessary)
                before iterations.

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_channels, n_bins, n_frames).
        """
        return super().__call__(input, n_iter=n_iter, initial_call=initial_call, **kwargs)

    def __repr__(self) -> str:
        s = "AuxIVA("
        s += "scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)


class GradIVA(GradIVABase):
    r"""Independent vector analysis (IVA) [#kim2006independent]_ using gradient descent.

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
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the gradient descent if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.

    Examples:
        Update demixing filters using Holonomic-type update:

        .. code-block:: python

            >>> def contrast_fn(y):
            ...     return 2 * np.linalg.norm(y, axis=1)

            >>> def score_fn(y):
            ...     norm = np.linalg.norm(y, axis=1, keepdims=True)
            ...     return y / np.maximum(norm, 1e-10)

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = GradIVA(
            ...     contrast_fn=contrast_fn,
            ...     score_fn=score_fn,
            ...     is_holonomic=True,
            ... )
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=5000)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters using Nonholonomic-type update:

        .. code-block:: python

            >>> def contrast_fn(y):
            ...     return 2 * np.linalg.norm(y, axis=1)

            >>> def score_fn(y):
            ...     norm = np.linalg.norm(y, axis=1, keepdims=True)
            ...     return y / np.maximum(norm, 1e-10)

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = GradIVA(
            ...     contrast_fn=contrast_fn,
            ...     score_fn=score_fn,
            ...     is_holonomic=False,
            ... )
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=5000)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

    .. [#kim2006independent]
        T. Kim, H. T. Attias, S.-Y. Lee, and T.-W. Lee,
        "Blind source separation exploiting higher-order frequency dependencies,"
        in *IEEE Trans. ASLP*, vol. 15, no. 1, pp. 70-79, 2007.
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
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            is_holonomic=is_holonomic,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
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


class NaturalGradIVA(GradIVABase):
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
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.

    Examples:
        Update demixing filters using Holonomic-type update:

        .. code-block:: python

            >>> def contrast_fn(y):
            ...     return 2 * np.linalg.norm(y, axis=1)

            >>> def score_fn(y):
            ...     norm = np.linalg.norm(y, axis=1, keepdims=True)
            ...     return y / np.maximum(norm, 1e-10)

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = NaturalGradIVA(
            ...     contrast_fn=contrast_fn,
            ...     score_fn=score_fn,
            ...     is_holonomic=True,
            ... )
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=500)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters using Nonholonomic-type update:

        .. code-block:: python

            >>> def contrast_fn(y):
            ...     return 2 * np.linalg.norm(y, axis=1)

            >>> def score_fn(y):
            ...     norm = np.linalg.norm(y, axis=1, keepdims=True)
            ...     return y / np.maximum(norm, 1e-10)

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = NaturalGradIVA(
            ...     contrast_fn=contrast_fn,
            ...     score_fn=score_fn,
            ...     is_holonomic=False,
            ... )
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=500)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)
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
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            is_holonomic=is_holonomic,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
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


class FastIVA(FastIVABase):
    r"""Fast independent vector analysis (Fast IVA) [#lee2007fast]_.

    Args:
        contrast_fn (callable):
            A contrast function which corresponds to :math:`-\log p(\vec{\boldsymbol{y}}_{jn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_frames).
        d_contrast_fn (callable):
            A derivative of the contrast function.
            This function is expected to receive (n_channels, n_frames)
            and return (n_channels, n_frames).
        dd_contrast_fn (callable):
            Second order derivative of the contrast function.
            This function is expected to receive (n_channels, n_frames)
            and return (n_channels, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.

    Examples:
        .. code-block:: python

            >>> from ssspy.transform import whiten
            >>> from ssspy.algorithm import projection_back

            >>> def contrast_fn(y):
            ...     return 2 * np.linalg.norm(y, axis=1)

            >>> def d_contrast_fn(y):
            ...     return 2 * np.ones_like(y)

            >>> def dd_contrast_fn(y):
            ...     return 2 * np.zeros_like(y)

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = FastIVA(
            ...     contrast_fn=contrast_fn,
            ...     d_contrast_fn=d_contrast_fn,
            ...     dd_contrast_fn=dd_contrast_fn,
            ...     scale_restoration=False,
            ... )
            >>> spectrogram_mix_whitened = whiten(spectrogram_mix)
            >>> spectrogram_est = iva(spectrogram_mix_whitened, n_iter=100)
            >>> spectrogram_est = projection_back(spectrogram_est, reference=spectrogram_mix)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

    .. [#lee2007fast] I. Lee et al.,
        "Fast fixed-point independent vector analysis algorithms \
        for convolutive blind source separation," *Signal Processing*,
        vol. 87, no. 8, pp. 1859-1871, 2007.
    """

    def __init__(
        self,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        d_contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        dd_contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["FastIVA"], None], List[Callable[["FastIVA"], None]]]
        ] = None,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

        if contrast_fn is None:
            raise ValueError("Specify contrast function.")
        else:
            self.contrast_fn = contrast_fn

        if d_contrast_fn is None:
            raise ValueError("Specify derivative of contrast function.")
        else:
            self.d_contrast_fn = d_contrast_fn

        if dd_contrast_fn is None:
            raise ValueError("Specify second order derivative of contrast function.")
        else:
            self.dd_contrast_fn = dd_contrast_fn

    def __call__(
        self, input: np.ndarray, n_iter: int = 100, initial_call: bool = True, **kwargs
    ) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                The number of iterations of demixing filter updates.
                Default: ``100``.
            initial_call (bool):
                If ``True``, perform callbacks (and computation of loss if necessary)
                before iterations.

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_channels, n_bins, n_frames).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        # Call __call__ of IVABase's parent, i.e. __call__ of IterativeMethodBase
        super(IVABase, self).__call__(n_iter=n_iter, initial_call=initial_call)

        if self.scale_restoration:
            self.restore_scale()

        self.output = self.separate(
            self.whitened_input, demix_filter=self.demix_filter, use_whitening=False
        )

        return self.output

    def __repr__(self) -> str:
        s = "FastIVA("
        s += "scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def update_once(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update demixing filters once.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Demixing filters are updated as follows:

        .. math::
            \boldsymbol{w}_{in}
            \leftarrow&\frac{1}{J}\sum_{j}
            \frac{G'_{\mathbb{R}}(\|\vec{\boldsymbol{y}}_{jn}\|_{2})}
            {2\|\vec{\boldsymbol{y}}_{jn}\|_{2}}
            \left(\boldsymbol{w}_{in}-y_{ijn}^{*}\boldsymbol{x}_{ij}\right) \notag \\
            &-\frac{1}{J}\sum_{j}\frac{|y_{ijn}|^{2}}{2\|\vec{\boldsymbol{y}}_{jn}\|_{2}}\left(
            \frac{G'_{\mathbb{R}}(\|\vec{\boldsymbol{y}}_{jn}\|_{2})}
            {\|\vec{\boldsymbol{y}}_{jn}\|_{2}}
            - G''_{\mathbb{R}}(\|\vec{\boldsymbol{y}}_{jn}\|_{2})
            \right)\boldsymbol{w}_{in} \\
            \boldsymbol{W}_{i}
            \leftarrow&\left(\boldsymbol{W}_{i}\boldsymbol{W}_{i}^{\mathsf{H}}\right)^{-\frac{1}{2}}
            \boldsymbol{W}_{i}.
        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        Z, W = self.whitened_input, self.demix_filter
        Y = self.separate(Z, demix_filter=W, use_whitening=False)

        norm = np.linalg.norm(Y, axis=1)
        varphi = self.d_contrast_fn(norm) / flooring_fn(2 * norm)  # (n_sources, n_frames)

        Y_conj = Y.conj()
        YZ = Y_conj[:, np.newaxis, :, :] * Z
        W_Hermite = W.transpose(1, 2, 0).conj()
        W_YZ = W_Hermite[:, :, :, np.newaxis] - YZ
        W_YZ = np.mean(varphi[:, np.newaxis, np.newaxis, :] * W_YZ, axis=-1)

        Y_GG = (2 * varphi - self.dd_contrast_fn(norm)) / flooring_fn(2 * norm)
        YY_GG = Y_GG[:, np.newaxis, :] * (np.abs(Y) ** 2)
        YY_GGW = np.mean(W_Hermite[:, :, :, np.newaxis] * YY_GG[:, np.newaxis, :, :], axis=-1)

        # Update
        W_Hermite = W_YZ - YY_GGW
        W = W_Hermite.transpose(2, 0, 1).conj()

        u, _, v_Hermite = np.linalg.svd(W)
        W = u @ v_Hermite

        self.demix_filter = W


class FasterIVA(FastIVABase):
    r"""Faster independent vector analysis (Faster IVA) [#brendel2021fasteriva]_.

    Args:
        contrast_fn (callable):
            A contrast function which corresponds to :math:`-\log p(\vec{\boldsymbol{y}}_{jn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_frames).
        d_contrast_fn (callable):
            A derivative of the contrast function.
            This function is expected to receive (n_channels, n_frames)
            and return (n_channels, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.

    Examples:
        .. code-block:: python

            >>> from ssspy.transform import whiten
            >>> from ssspy.algorithm import projection_back

            >>> def contrast_fn(y):
            ...     return 2 * np.linalg.norm(y, axis=1)

            >>> def d_contrast_fn(y):
            ...     return 2 * np.ones_like(y)

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = FasterIVA(
            ...     contrast_fn=contrast_fn,
            ...     d_contrast_fn=d_contrast_fn,
            ...     scale_restoration=False,
            ... )
            >>> spectrogram_mix_whitened = whiten(spectrogram_mix)
            >>> spectrogram_est = iva(spectrogram_mix_whitened, n_iter=100)
            >>> spectrogram_est = projection_back(spectrogram_est, reference=spectrogram_mix)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

    .. [#brendel2021fasteriva] A. Brendel and W. Kellermann,
        "Faster IVA: Update rules for independent vector analysis based on negentropy \
        and the majorize-minimize principle,"
        in *Proc. WASPAA*, pp. 131-135, 2021.
    """

    def __init__(
        self,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        d_contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["FasterIVA"], None], List[Callable[["FasterIVA"], None]]]
        ] = None,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )
        if contrast_fn is None:
            raise ValueError("Specify contrast function.")
        else:
            self.contrast_fn = contrast_fn

        if d_contrast_fn is None:
            raise ValueError("Specify derivative of contrast function.")
        else:
            self.d_contrast_fn = d_contrast_fn

    def __call__(
        self, input: np.ndarray, n_iter: int = 100, initial_call: bool = True, **kwargs
    ) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                The number of iterations of demixing filter updates.
                Default: ``100``.
            initial_call (bool):
                If ``True``, perform callbacks (and computation of loss if necessary)
                before iterations.

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_channels, n_bins, n_frames).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        # Call __call__ of IVABase's parent, i.e. __call__ of IterativeMethodBase
        super(IVABase, self).__call__(n_iter=n_iter, initial_call=initial_call)

        if self.scale_restoration:
            self.restore_scale()

        self.output = self.separate(
            self.whitened_input, demix_filter=self.demix_filter, use_whitening=False
        )

        return self.output

    def __repr__(self) -> str:
        s = "FasterIVA("
        s += "scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def update_once(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update demixing filters once.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        In FasterIVA, we compute the eigenvector of :math:`\boldsymbol{U}_{in}`
        which corresponds to the largest eigenvalue by solving

        .. math::
            \boldsymbol{U}_{in}\boldsymbol{w}_{in}
            = \lambda_{in}\boldsymbol{w}_{in}.

        Then,

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\left(\boldsymbol{W}_{i}\boldsymbol{W}_{i}^{\mathsf{H}}\right)^{-\frac{1}{2}}
            \boldsymbol{W}_{i}.
        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        Z, W = self.whitened_input, self.demix_filter
        Y = self.separate(Z, demix_filter=W, use_whitening=False)

        ZZ_Hermite = Z[:, np.newaxis, :, :] * Z[np.newaxis, :, :, :].conj()
        ZZ_Hermite = ZZ_Hermite.transpose(2, 0, 1, 3)  # (n_bins, n_channels, n_channels, n_frames)
        norm = np.linalg.norm(Y, axis=1)
        varphi = self.d_contrast_fn(norm) / flooring_fn(2 * norm)  # (n_sources, n_frames)
        varphi_ZZ = varphi[:, np.newaxis, np.newaxis, :] * ZZ_Hermite[:, np.newaxis, :, :, :]
        U = np.mean(varphi_ZZ, axis=-1)  # (n_bins, n_sources, n_channels, n_channels)

        _, w = eigh(U)  # (n_bins, n_sources, n_channels, n_channels)
        W = w[..., -1].conj()  # eigenvector that corresponds to largest eigenvalue
        u, _, v_Hermite = np.linalg.svd(W)
        W = u @ v_Hermite

        self.demix_filter = W


class AuxIVA(AuxIVABase):
    r"""Auxiliary-function-based independent vector analysis (IVA) [#ono2011stable]_.

    Args:
        spatial_algorithm (str):
            Algorithm for demixing filter updates.
            Choose ``IP``, ``IP1``, ``IP2``, ``ISS``, ``ISS1``, ``ISS2``, or ``IPA``.
            Default: ``IP``.
        contrast_fn (callable):
            A contrast function which corresponds to :math:`-\log p(\vec{\boldsymbol{y}}_{jn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_frames).
        d_contrast_fn (callable):
            A derivative of the contrast function.
            This function is expected to receive (n_channels, n_frames)
            and return (n_channels, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
        pair_selector (callable, optional):
            Selector to choose updaing pair in ``IP2`` and ``ISS2``.
            If ``None`` is given, ``sequential_pair_selector`` is used.
            Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the demixing filter update if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.
        lqpqm_normalization (bool):
            This keyword argument can be specified when ``spatial_algorithm='IPA'``.
            If ``True``, normalization by trace is applied to positive semi-definite matrix
            in LQPQM. Default: ``True``.
        newton_iter (int):
            This keyword argument can be specified when ``spatial_algorithm='IPA'``.
            Number of iterations in Newton method. Default: ``1``.

    Examples:
        Update demixing filters by IP:

        .. code-block:: python

            >>> def contrast_fn(y):
            ...     return 2 * np.linalg.norm(y, axis=1)

            >>> def d_contrast_fn(y):
            ...     return 2 * np.ones_like(y)

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = AuxIVA(
            ...     spatial_algorithm="IP",
            ...     contrast_fn=contrast_fn,
            ...     d_contrast_fn=d_contrast_fn,
            ... )
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by IP2:

        .. code-block:: python

            >>> from ssspy.utils.select_pair import sequential_pair_selector

            >>> def contrast_fn(y):
            ...     return 2 * np.linalg.norm(y, axis=1)

            >>> def d_contrast_fn(y):
            ...     return 2 * np.ones_like(y)

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = AuxIVA(
            ...     spatial_algorithm="IP2",
            ...     contrast_fn=contrast_fn,
            ...     d_contrast_fn=d_contrast_fn,
            ...     pair_selector=sequential_pair_selector,
            ... )
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by ISS:

        .. code-block:: python

            >>> def contrast_fn(y):
            ...     return 2 * np.linalg.norm(y, axis=1)

            >>> def d_contrast_fn(y):
            ...     return 2 * np.ones_like(y)

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = AuxIVA(
            ...     spatial_algorithm="ISS",
            ...     contrast_fn=contrast_fn,
            ...     d_contrast_fn=d_contrast_fn,
            ... )
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by ISS2:

        .. code-block:: python

            >>> import functools
            >>> from ssspy.utils.select_pair import sequential_pair_selector

            >>> def contrast_fn(y):
            ...     return 2 * np.linalg.norm(y, axis=1)

            >>> def d_contrast_fn(y):
            ...     return 2 * np.ones_like(y)

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = AuxIVA(
            ...     spatial_algorithm="ISS2",
            ...     contrast_fn=contrast_fn,
            ...     d_contrast_fn=d_contrast_fn,
            ...     pair_selector=functools.partial(sequential_pair_selector, step=2),
            ... )
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by IPA:

        .. code-block:: python

            >>> def contrast_fn(y):
            ...     return 2 * np.linalg.norm(y, axis=1)

            >>> def d_contrast_fn(y):
            ...     return 2 * np.ones_like(y)

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = AuxIVA(
            ...     spatial_algorithm="IPA",
            ...     contrast_fn=contrast_fn,
            ...     d_contrast_fn=d_contrast_fn,
            ... )
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

    .. [#ono2011stable]
        N. Ono,
        "Stable and fast update rules for independent vector analysis based on \
        auxiliary function technique,"
        in *Proc. WASPAA*, 2011, p.189-192.
    """

    _ipa_default_kwargs = {"lqpqm_normalization": True, "newton_iter": 1}
    _default_kwargs = _ipa_default_kwargs

    def __init__(
        self,
        spatial_algorithm: str = "IP",
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        d_contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        pair_selector: Optional[Callable[[int], Iterable[Tuple[int, int]]]] = None,
        callbacks: Optional[
            Union[Callable[["AuxIVA"], None], List[Callable[["AuxIVA"], None]]]
        ] = None,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            contrast_fn=contrast_fn,
            d_contrast_fn=d_contrast_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

        assert spatial_algorithm in spatial_algorithms, "Not support {}.".format(spatial_algorithm)

        self.spatial_algorithm = spatial_algorithm

        if pair_selector is None:
            if spatial_algorithm in ["IP2", "ISS2"]:
                self.pair_selector = sequential_pair_selector
        else:
            self.pair_selector = pair_selector

        if spatial_algorithm == "IPA":
            valid_keys = set(self.__class__._ipa_default_kwargs.keys())
        else:
            valid_keys = set()

        invalid_keys = set(kwargs) - valid_keys

        assert invalid_keys == set(), "Invalid keywords {} are given.".format(invalid_keys)

        for key, value in kwargs.items():
            setattr(self, key, value)

        # set default values if necessary
        for key in valid_keys:
            if not hasattr(self, key):
                value = self.__class__._default_kwargs[key]
                setattr(self, key, value)

    def __call__(
        self, input: np.ndarray, n_iter: int = 100, initial_call: bool = True, **kwargs
    ) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                The number of iterations of demixing filter updates.
                Default: ``100``.
            initial_call (bool):
                If ``True``, perform callbacks (and computation of loss if necessary)
                before iterations.

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_channels, n_bins, n_frames).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        # Call __call__ of IVABase's parent, i.e. __call__ of IterativeMethodBase
        super(IVABase, self).__call__(n_iter=n_iter, initial_call=initial_call)

        if self.scale_restoration:
            self.restore_scale()

        if self.demix_filter is None:
            pass
        else:
            self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "AuxIVA("
        s += "spatial_algorithm={spatial_algorithm}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(self, **kwargs) -> None:
        r"""Reset attributes by given keyword arguments.

        Args:
            kwargs:
                Keyword arguments to set as attributes of IVA.
        """
        super()._reset(**kwargs)

        if self.spatial_algorithm in ["ISS", "ISS1", "ISS2", "IPA"]:
            self.demix_filter = None

    def update_once(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update demixing filters once.

        - If ``self.spatial_algorithm`` is ``IP`` or ``IP1``, ``update_once_ip1`` is called.
        - If ``self.spatial_algorithm`` is ``IP2``, ``update_once_ip2`` is called.
        - If ``self.spatial_algorithm`` is ``ISS`` or ``ISS1``, ``update_once_iss1`` is called.
        - If ``self.spatial_algorithm`` is ``ISS2``, ``update_once_iss2`` is called.
        - If ``self.spatial_algorithm`` is ``IPA``, ``update_once_ipa`` is called.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.spatial_algorithm in ["IP", "IP1"]:
            self.update_once_ip1(flooring_fn=flooring_fn)
        elif self.spatial_algorithm in ["IP2"]:
            self.update_once_ip2(flooring_fn=flooring_fn)
        elif self.spatial_algorithm in ["ISS", "ISS1"]:
            self.update_once_iss1(flooring_fn=flooring_fn)
        elif self.spatial_algorithm in ["ISS2"]:
            self.update_once_iss2(flooring_fn=flooring_fn)
        elif self.spatial_algorithm in ["IPA"]:
            self.update_once_ipa(flooring_fn=flooring_fn)
        else:
            raise NotImplementedError("Not support {}.".format(self.spatial_algorithm))

    def update_once_ip1(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update demixing filters once using iterative projection.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Compute auxiliary variables:

        .. math::
            \bar{r}_{jn}
            \leftarrow\|\vec{\boldsymbol{y}}_{jn}\|_{2}

        Then, demixing filters are updated sequentially for :math:`n=1,\ldots,N` as follows:

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
            \varphi(\bar{r}_{jn})\boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}, \\
            \varphi(\bar{r}_{jn})
            &= \frac{G'_{\mathbb{R}}(\bar{r}_{jn})}{2\bar{r}_{jn}}, \\
            G(\vec{\boldsymbol{y}}_{jn})
            &= -\log p(\vec{\boldsymbol{y}}_{jn}), \\
            G_{\mathbb{R}}(\|\vec{\boldsymbol{y}}_{jn}\|_{2})
            &= G(\vec{\boldsymbol{y}}_{jn}).
        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        XX_Hermite = X[:, np.newaxis, :, :] * X[np.newaxis, :, :, :].conj()
        XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)  # (n_bins, n_channels, n_channels, n_frames)
        norm = np.linalg.norm(Y, axis=1)
        denom = flooring_fn(2 * norm)
        weight = self.d_contrast_fn(norm) / denom  # (n_sources, n_frames)
        GXX = weight[:, np.newaxis, np.newaxis, :] * XX_Hermite[:, np.newaxis, :, :, :]
        U = np.mean(GXX, axis=-1)  # (n_bins, n_sources, n_channels, n_channels)

        self.demix_filter = update_by_ip1(W, U, flooring_fn=flooring_fn)

    def update_once_ip2(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update demixing filters once using pairwise iterative projection.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        For :math:`n_{1}` and :math:`n_{2}` (:math:`n_{1}\neq n_{2}`),
        compute auxiliary variables:

        .. math::
            \bar{r}_{jn_{1}}
            &\leftarrow\|\vec{\boldsymbol{y}}_{jn_{1}}\|_{2} \\
            \bar{r}_{jn_{2}}
            &\leftarrow\|\vec{\boldsymbol{y}}_{jn_{2}}\|_{2}

        Then, for :math:`n=n_{1},n_{2}`, compute weighted covariance matrix as follows:

        .. math::
            \boldsymbol{U}_{in_{1}}
            &= \frac{1}{J}\sum_{j}
            \varphi(\bar{r}_{jn_{1}})\boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}, \\
            \boldsymbol{U}_{in_{2}}
            &= \frac{1}{J}\sum_{j}
            \varphi(\bar{r}_{jn_{2}})\boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}},

        where

        .. math::
            \varphi(\bar{r}_{jn})
            = \frac{G'_{\mathbb{R}}(\bar{r}_{jn})}{2\bar{r}_{jn}}.

        Using :math:`\boldsymbol{U}_{in_{1}}` and
        :math:`\boldsymbol{U}_{in_{2}}`, we compute generalized eigenvectors.

        .. math::
            \left({\boldsymbol{P}_{in_{1}}^{(n_{1},n_{2})}}^{\mathsf{H}}\boldsymbol{U}_{in_{1}}
            \boldsymbol{P}_{in_{1}}^{(n_{1},n_{2})}\right)\boldsymbol{h}_{i}
            = \lambda_{i}
            \left({\boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}}^{\mathsf{H}}\boldsymbol{U}_{in_{2}}
            \boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}\right)\boldsymbol{h}_{i},

        where

        .. math::
            \boldsymbol{P}_{in_{1}}^{(n_{1},n_{2})}
            &= (\boldsymbol{W}_{i}\boldsymbol{U}_{in_{1}})^{-1}
            (
            \begin{array}{cc}
                \boldsymbol{e}_{n_{1}} & \boldsymbol{e}_{n_{2}}
            \end{array}
            ), \\
            \boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}
            &= (\boldsymbol{W}_{i}\boldsymbol{U}_{in_{2}})^{-1}
            (
            \begin{array}{cc}
                \boldsymbol{e}_{n_{1}} & \boldsymbol{e}_{n_{2}}
            \end{array}
            ).

        After that, we standardize two eigenvectors :math:`\boldsymbol{h}_{in_{1}}`
        and :math:`\boldsymbol{h}_{in_{2}}`.

        .. math::
            \boldsymbol{h}_{in_{1}}
            &\leftarrow\frac{\boldsymbol{h}_{in_{1}}}
            {\sqrt{\boldsymbol{h}_{in_{1}}^{\mathsf{H}}
            \left({\boldsymbol{P}_{in_{1}}^{(n_{1},n_{2})}}^{\mathsf{H}}\boldsymbol{U}_{in_{1}}
            \boldsymbol{P}_{in_{1}}^{(n_{1},n_{2})}\right)
            \boldsymbol{h}_{in_{1}}}}, \\
            \boldsymbol{h}_{in_{2}}
            &\leftarrow\frac{\boldsymbol{h}_{in_{2}}}
            {\sqrt{\boldsymbol{h}_{in_{2}}^{\mathsf{H}}
            \left({\boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}}^{\mathsf{H}}\boldsymbol{U}_{in_{2}}
            \boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}\right)
            \boldsymbol{h}_{in_{2}}}}.

        Then, update :math:`\boldsymbol{w}_{in_{1}}` and :math:`\boldsymbol{w}_{in_{2}}`
        simultaneously.

        .. math::
            \boldsymbol{w}_{in_{1}}
            &\leftarrow \boldsymbol{P}_{in_{1}}^{(n_{1},n_{2})}\boldsymbol{h}_{in_{1}} \\
            \boldsymbol{w}_{in_{2}}
            &\leftarrow \boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}\boldsymbol{h}_{in_{2}}.

        At each iteration, we update pairs of :math:`n_{1}` and :math:`n_{1}`
        for :math:`n_{1}\neq n_{2}`.
        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        n_sources = self.n_sources
        X, W = self.input, self.demix_filter
        XX_Hermite = X[:, np.newaxis, :, :] * X[np.newaxis, :, :, :].conj()
        XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)

        for m, n in self.pair_selector(n_sources):
            W_mn = W[:, (m, n), :]
            Y_mn = self.separate(X, demix_filter=W_mn)

            norm = np.linalg.norm(Y_mn, axis=1)
            weight = self.d_contrast_fn(norm) / flooring_fn(2 * norm)
            GXX_mn = weight[:, np.newaxis, np.newaxis, :] * XX_Hermite[:, np.newaxis, :, :, :]
            U_mn = np.mean(GXX_mn, axis=-1)

            W[:, (m, n), :] = update_by_ip2_one_pair(
                W,
                U_mn,
                pair=(m, n),
                flooring_fn=flooring_fn,
            )

        self.demix_filter = W

    def update_once_iss1(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update estimated spectrograms once using \
        iterative source steering [#scheibler2020fast]_.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        First, update auxiliary variables

        .. math::
            \bar{r}_{jn}
            \leftarrow\|\vec{\boldsymbol{y}}_{jn}\|_{2}.

        Then, update :math:`y_{ijn}` as follows:

        .. math::
            \boldsymbol{y}_{ij}
            & \leftarrow\boldsymbol{y}_{ij} - \boldsymbol{d}_{in}y_{ijn}, \\
            d_{inn'}
            &= \begin{cases}
                \dfrac{\sum_{j}\dfrac{G'_{\mathbb{R}}(\bar{r}_{jn'})}{2\bar{r}_{jn'}}
                y_{ijn'}y_{ijn}^{*}}{\sum_{j}\dfrac{G'_{\mathbb{R}}(\bar{r}_{jn'})}
                {2\bar{r}_{jn'}}|y_{ijn}|^{2}}
                & (n'\neq n) \\
                1 - \dfrac{1}{\sqrt{\dfrac{1}{J}\sum_{j}\dfrac{G'_{\mathbb{R}}(\bar{r}_{jn'})}
                {2\bar{r}_{jn'}}
                |y_{ijn}|^{2}}} & (n'=n)
            \end{cases}.

        .. [#scheibler2020fast] R. Scheibler and N. Ono,
            "Fast and stable blind source separation with rank-1 updates,"
            in *Proc. ICASSP*, 2020, pp. 236-240.
        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        Y = self.output
        r = np.linalg.norm(Y, axis=1)
        denom = flooring_fn(2 * r)
        varphi = self.d_contrast_fn(r) / denom  # (n_sources, n_frames)

        self.output = update_by_iss1(Y, varphi[:, np.newaxis, :], flooring_fn=flooring_fn)

    def update_once_iss2(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update estimated spectrograms once using \
        pairwise iterative source steering [#ikeshita2022iss2]_.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        First, we compute auxiliary variables:

        .. math::
            \bar{r}_{jn}
            \leftarrow\|\vec{\boldsymbol{y}}_{jn}\|_{2},

        where

        .. math::
            G(\vec{\boldsymbol{y}}_{jn})
            &= -\log p(\vec{\boldsymbol{y}}_{jn}), \\
            G_{\mathbb{R}}(\|\vec{\boldsymbol{y}}_{jn}\|_{2})
            &= G(\vec{\boldsymbol{y}}_{jn}).

        Then, we compute :math:`\boldsymbol{G}_{in}^{(n_{1},n_{2})}` \
        and :math:`\boldsymbol{f}_{in}^{(n_{1},n_{2})}` for :math:`n_{1}\neq n_{2}`:

        .. math::
            \begin{array}{rclc}
                \boldsymbol{G}_{in}^{(n_{1},n_{2})}
                &=& {\displaystyle\frac{1}{J}\sum_{j}}\varphi(\bar{r}_{jn})
                \boldsymbol{y}_{ij}^{(n_{1},n_{2})}{\boldsymbol{y}_{ij}^{(n_{1},n_{2})}}^{\mathsf{H}}
                &(n=1,\ldots,N), \\
                \boldsymbol{f}_{in}^{(n_{1},n_{2})}
                &=& {\displaystyle\frac{1}{J}\sum_{j}}
                \varphi(\bar{r}_{jn})y_{ijn}^{*}\boldsymbol{y}_{ij}^{(n_{1},n_{2})}
                &(n\neq n_{1},n_{2}), \\
                \varphi(\bar{r}_{jn})
                &=&\dfrac{G'_{\mathbb{R}}(\bar{r}_{jn})}{2\bar{r}_{jn}}.
            \end{array}

        Using :math:`\boldsymbol{G}_{in}^{(n_{1},n_{2})}` and \
        :math:`\boldsymbol{f}_{in}^{(n_{1},n_{2})}`, we compute

        .. math::
            \begin{array}{rclc}
                \boldsymbol{p}_{in}
                &=& \dfrac{\boldsymbol{h}_{in}}
                {\sqrt{\boldsymbol{h}_{in}^{\mathsf{H}}\boldsymbol{G}_{in}^{(n_{1},n_{2})}
                \boldsymbol{h}_{in}}} & (n=n_{1},n_{2}), \\
                \boldsymbol{q}_{in}
                &=& -{\boldsymbol{G}_{in}^{(n_{1},n_{2})}}^{-1}\boldsymbol{f}_{in}^{(n_{1},n_{2})}
                & (n\neq n_{1},n_{2}),
            \end{array}

        where :math:`\boldsymbol{h}_{in}` (:math:`n=n_{1},n_{2}`) is \
        a generalized eigenvector obtained from

        .. math::
            \boldsymbol{G}_{in_{1}}^{(n_{1},n_{2})}\boldsymbol{h}_{i}
            = \lambda_{i}\boldsymbol{G}_{in_{2}}^{(n_{1},n_{2})}\boldsymbol{h}_{i}.

        Separated signal :math:`y_{ijn}` is updated as follows:

        .. math::
            y_{ijn}
            &\leftarrow\begin{cases}
            &\boldsymbol{p}_{in}^{\mathsf{H}}\boldsymbol{y}_{ij}^{(n_{1},n_{2})}
            & (n=n_{1},n_{2}) \\
            &\boldsymbol{q}_{in}^{\mathsf{H}}\boldsymbol{y}_{ij}^{(n_{1},n_{2})} + y_{ijn}
            & (n\neq n_{1},n_{2})
            \end{cases}.

        .. [#ikeshita2022iss2]
            R. Ikeshita and T. Nakatani,
            "ISS2: An extension of iterative source steering algorithm for \
            majorization-minimization-based independent vector analysis,"
            *arXiv:2202.00875*, 2022.
        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        Y = self.output

        # Auxiliary variables
        r = np.linalg.norm(Y, axis=1)
        varphi = self.d_contrast_fn(r) / flooring_fn(2 * r)

        self.output = update_by_iss2(
            Y,
            varphi[:, np.newaxis, :],
            flooring_fn=flooring_fn,
            pair_selector=self.pair_selector,
        )

    def update_once_ipa(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update estimated spectrograms once using \
        iterative projection with adjustment [#scheibler2021independent]_.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        First, we compute auxiliary variables:

        .. math::
            \bar{r}_{jn}
            \leftarrow\|\vec{\boldsymbol{y}}_{jn}\|_{2},

        where

        .. math::
            G(\vec{\boldsymbol{y}}_{jn})
            &= -\log p(\vec{\boldsymbol{y}}_{jn}), \\
            G_{\mathbb{R}}(\|\vec{\boldsymbol{y}}_{jn}\|_{2})
            &= G(\vec{\boldsymbol{y}}_{jn}).

        Then, by defining, :math:`\tilde{\boldsymbol{U}}_{in'}`,
        :math:`\boldsymbol{A}_{in}\in\mathbb{R}^{(N-1)\times(N-1)}`,
        :math:`\boldsymbol{b}_{in}\in\mathbb{C}^{N-1}`,
        :math:`\boldsymbol{C}_{in}\in\mathbb{C}^{(N-1)\times(N-1)}`,
        :math:`\boldsymbol{d}_{in}\in\mathbb{C}^{N-1}`,
        and :math:`z_{in}\in\mathbb{R}_{\geq 0}` as follows:

        .. math::

            \tilde{\boldsymbol{U}}_{in'}
            &= \frac{1}{J}\sum_{j}\frac{G'_{\mathbb{R}}(\bar{r}_{jn'})}{2\bar{r}_{jn'}}
            \boldsymbol{y}_{ij}\boldsymbol{y}_{ij}^{\mathsf{H}}, \\
            \boldsymbol{A}_{in}
            &= \mathrm{diag}(\ldots,
            \boldsymbol{e}_{n}^{\mathsf{T}}\tilde{\boldsymbol{U}}_{in'}\boldsymbol{e}_{n}
            ,\ldots)~~(n'\neq n), \\
            \boldsymbol{b}_{in}
            &= (\ldots,
            \boldsymbol{e}_{n}^{\mathsf{T}}\tilde{\boldsymbol{U}}_{in'}\boldsymbol{e}_{n'}
            ,\ldots)^{\mathsf{T}}~~(n'\neq n), \\
            \boldsymbol{C}_{in}
            &= \bar{\boldsymbol{E}}_{n}^{\mathsf{T}}(\tilde{\boldsymbol{U}}_{in}^{-1})^{*}
            \bar{\boldsymbol{E}}_{n}, \\
            \boldsymbol{d}_{in}
            &= \bar{\boldsymbol{E}}_{n}^{\mathsf{T}}(\tilde{\boldsymbol{U}}_{in}^{-1})^{*}
            \boldsymbol{e}_{n}, \\
            z_{in}
            &= \boldsymbol{e}_{n}^{\mathsf{T}}\tilde{\boldsymbol{U}}_{in}^{-1}\boldsymbol{e}_{n}
            - \boldsymbol{d}_{in}^{\mathsf{H}}\boldsymbol{C}_{in}^{-1}\boldsymbol{d}_{in},

        :math:`\boldsymbol{y}_{ij}` is updated via log-quadratically penelized
        quadratic minimization (LQPQM).

        .. math::
            \check{\boldsymbol{q}}_{in}
            &\leftarrow \mathrm{LQPQM2}(\boldsymbol{H}_{in},\boldsymbol{v}_{in},z_{in}), \\
            \boldsymbol{q}_{in}
            &\leftarrow \boldsymbol{G}_{in}^{-1}\check{\boldsymbol{q}}_{in}
            - \boldsymbol{A}_{in}^{-1}\boldsymbol{b}_{in}, \\
            \tilde{\boldsymbol{q}}_{in}
            &\leftarrow \boldsymbol{e}_{n} - \bar{\boldsymbol{E}}_{n}\boldsymbol{q}_{in}, \\
            \boldsymbol{p}_{in}
            &\leftarrow \frac{\tilde{\boldsymbol{U}}_{in}^{-1}\tilde{\boldsymbol{q}}_{in}^{*}}
            {\sqrt{(\tilde{\boldsymbol{q}}_{in}^{*})^{\mathsf{H}}\tilde{\boldsymbol{U}}_{in}^{-1}
            \tilde{\boldsymbol{q}}_{in}^{*}}}, \\
            \boldsymbol{\Upsilon}_{i}^{(n)}
            &\leftarrow \boldsymbol{I}
            + \boldsymbol{e}_{n}(\boldsymbol{p}_{in} - \boldsymbol{e}_{n})^{\mathsf{H}}
            + \bar{\boldsymbol{E}}_{n}\boldsymbol{q}_{in}^{*}\boldsymbol{e}_{n}^{\mathsf{T}}, \\
            \boldsymbol{y}_{ij}
            &\leftarrow \boldsymbol{\Upsilon}_{i}^{(n)}\boldsymbol{y}_{ij},

        .. [#scheibler2021independent]
            R. Scheibler,
            "Independent vector analysis via log-quadratically penalized quadratic minimization,"
            *IEEE Trans. Signal Processing*, vol. 69, pp. 2509-2524, 2021.

        """
        self.lqpqm_normalization: bool
        self.newton_iter: int

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        Y = self.output
        r = np.linalg.norm(Y, axis=1)
        denom = flooring_fn(2 * r)
        varphi = self.d_contrast_fn(r) / denom

        normalization = self.lqpqm_normalization
        max_iter = self.newton_iter

        self.output = update_by_ipa(
            Y,
            varphi[:, np.newaxis, :],
            normalization=normalization,
            flooring_fn=flooring_fn,
            max_iter=max_iter,
        )

    def compute_loss(self) -> float:
        r"""Compute loss."""
        if self.demix_filter is None:
            X, Y = self.input, self.output
            G = self.contrast_fn(Y)  # (n_sources, n_frames)
            X, Y = X.transpose(1, 0, 2), Y.transpose(1, 0, 2)
            X_Hermite = X.transpose(0, 2, 1).conj()
            XX_Hermite = X @ X_Hermite  # (n_bins, n_channels, n_channels)
            W = Y @ X_Hermite @ np.linalg.inv(XX_Hermite)
            logdet = self.compute_logdet(W)  # (n_bins,)
            loss = np.sum(np.mean(G, axis=1), axis=0) - 2 * np.sum(logdet, axis=0)
            loss = loss.item()

            return loss
        else:
            return super().compute_loss()

    def apply_projection_back(self) -> None:
        r"""Apply projection back technique to estimated spectrograms."""
        if self.demix_filter is None:
            assert self.scale_restoration, "Set self.scale_restoration=True."

            X, Y = self.input, self.output
            Y_scaled = projection_back(Y, reference=X, reference_id=self.reference_id)

            self.output = Y_scaled
        else:
            super().apply_projection_back()

    def apply_minimal_distortion_principle(self) -> None:
        r"""Apply minimal distortion principle to estimated spectrograms."""
        if self.demix_filter is None:
            X, Y = self.input, self.output
            Y_scaled = minimal_distortion_principle(Y, reference=X, reference_id=self.reference_id)

            self.output = Y_scaled
        else:
            super().apply_minimal_distortion_principle()


class PDSIVA(PDSBSS):
    def __init__(
        self,
        mu1: float = 1,
        mu2: float = 1,
        alpha: float = None,
        relaxation: float = 1,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        prox_penalty: Callable[[np.ndarray, float], np.ndarray] = None,
        callbacks: Optional[
            Union[Callable[["PDSIVA"], None], List[Callable[["PDSIVA"], None]]]
        ] = None,
        scale_restoration: bool = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        if contrast_fn is not None and prox_penalty is None:
            raise ValueError("Set prox_penalty.")
        elif contrast_fn is None and prox_penalty is not None:
            raise ValueError("Set contrast_fn.")
        elif contrast_fn is None and prox_penalty is None:

            def _contrast_fn(y: np.ndarray) -> np.ndarray:
                return np.linalg.norm(y, axis=1)

            def _prox_penalty(x: np.ndarray, step_size: float = 1) -> np.ndarray:
                return prox.l21(x, step_size=step_size, axis2=1)

            contrast_fn = _contrast_fn
            prox_penalty = _prox_penalty

        def penalty_fn(y: np.ndarray) -> float:
            r"""Sum of contrast function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                Computed loss.
            """
            G = contrast_fn(y)  # (n_sources, n_frames)
            loss = np.sum(G, axis=(0, 1))
            loss = loss.item()

            return loss

        super().__init__(
            mu1=mu1,
            mu2=mu2,
            alpha=alpha,
            relaxation=relaxation,
            penalty_fn=penalty_fn,
            prox_penalty=prox_penalty,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

        self.contrast_fn = contrast_fn


class ADMMIVA(ADMMBSS):
    def __init__(
        self,
        rho: float = 1,
        alpha: float = None,
        relaxation: float = 1,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        prox_penalty: Callable[[np.ndarray, float], np.ndarray] = None,
        callbacks: Optional[
            Union[Callable[["ADMMIVA"], None], List[Callable[["ADMMIVA"], None]]]
        ] = None,
        scale_restoration: bool = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        if contrast_fn is not None and prox_penalty is None:
            raise ValueError("Set prox_penalty.")
        elif contrast_fn is None and prox_penalty is not None:
            raise ValueError("Set contrast_fn.")
        elif contrast_fn is None and prox_penalty is None:

            def _contrast_fn(y: np.ndarray) -> np.ndarray:
                return np.linalg.norm(y, axis=1)

            def _prox_penalty(x: np.ndarray, step_size: float = 1) -> np.ndarray:
                return prox.l21(x, step_size=step_size, axis2=1)

            contrast_fn = _contrast_fn
            prox_penalty = _prox_penalty

        def penalty_fn(y: np.ndarray) -> float:
            r"""Sum of contrast function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                Computed loss.
            """
            G = contrast_fn(y)  # (n_sources, n_frames)
            loss = np.sum(G, axis=(0, 1))
            loss = loss.item()

            return loss

        super().__init__(
            rho=rho,
            alpha=alpha,
            relaxation=relaxation,
            penalty_fn=penalty_fn,
            prox_penalty=prox_penalty,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

        self.contrast_fn = contrast_fn


class GradLaplaceIVA(GradIVA):
    r"""Independent vector analysis (IVA) using the gradient descent on a Laplace distribution.

    We assume :math:`\vec{\boldsymbol{y}}_{jn}` follows a Laplace distribution.

    .. math::
        p(\vec{\boldsymbol{y}}_{jn})\propto\exp(\|\vec{\boldsymbol{y}}_{jn}\|_{2})

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the gradient descent if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.

    Examples:
        Update demixing filters using Holonomic-type update:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = GradLaplaceIVA(is_holonomic=True)
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=5000)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters using Nonholonomic-type update:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = GradLaplaceIVA(is_holonomic=False)
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=5000)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)
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
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        def contrast_fn(y: np.ndarray) -> np.ndarray:
            r"""Contrast function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray of the shape is (n_sources, n_frames).
            """
            return 2 * np.linalg.norm(y, axis=1)

        def score_fn(y: np.ndarray) -> np.ndarray:
            r"""Score function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray of the shape is (n_sources, n_bins, n_frames).
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
            scale_restoration=scale_restoration,
            record_loss=record_loss,
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

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L} \
            = \frac{2}{J}\sum_{j,n}\|\vec{\boldsymbol{y}}_{jn}\|_{2} \
            - 2\sum_{i}\log|\det\boldsymbol{W}_{i}|.

        Returns:
            Computed loss.
        """
        return super().compute_loss()


class GradGaussIVA(GradIVA):
    r"""Independent vector analysis (IVA) using the gradient descent on \
    a time-varying Gaussian distribution.

    We assume :math:`\vec{\boldsymbol{y}}_{jn}` follows a time-varying Gaussian distribution.

    .. math::
        p(\vec{\boldsymbol{y}}_{jn})
        \propto\frac{1}{\alpha_{jn}^{I}}
        \exp\left(\frac{\|\vec{\boldsymbol{y}}_{jn}\|_{2}^{2}}{\alpha_{jn}}\right).

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the gradient descent if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.

    Examples:
        Update demixing filters using Holonomic-type update:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = GradGaussIVA(is_holonomic=True)
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=5000)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters using Nonholonomic-type update:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = GradGaussIVA(is_holonomic=False)
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=5000)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["GradGaussIVA"], None], List[Callable[["GradGaussIVA"], None]]]
        ] = None,
        is_holonomic: bool = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        def contrast_fn(y: np.ndarray) -> np.ndarray:
            r"""
            Args:
                y (numpy.ndarray):
                    Separated signal with shape of (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray of computed contrast function.
                The shape is (n_sources, n_frames).
            """
            n_bins = self.n_bins
            alpha = self.variance
            norm = np.linalg.norm(y, axis=1)

            return n_bins * np.log(alpha) + (norm**2) / alpha

        def score_fn(y: np.ndarray) -> np.ndarray:
            r"""
            Args:
                y (numpy.ndarray):
                    Norm of separated signal.
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray of computed contrast function.
                The shape is (n_sources, n_frames).
            """
            alpha = self.variance
            return y / alpha[:, np.newaxis, :]

        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            is_holonomic=is_holonomic,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

    def _reset(self, **kwargs) -> None:
        r"""Reset attributes by given keyword arguments.

        We also set variance of Gaussian distribution.

        Args:
            kwargs:
                Keyword arguments to set as attributes of IVA.
        """
        super()._reset(**kwargs)

        n_sources, n_frames = self.n_sources, self.n_frames

        self.variance = np.ones((n_sources, n_frames))

    def update_once(self) -> None:
        r"""Update variance and demixing filters and once."""
        self.update_source_model()

        super().update_once()

    def update_source_model(self) -> None:
        r"""Update variance of Gaussian distribution."""
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        self.variance = np.mean(np.abs(Y) ** 2, axis=1)


class NaturalGradLaplaceIVA(NaturalGradIVA):
    r"""Independent vector analysis (IVA) using the natural gradient descent \
    on a Laplace distribution.

    We assume :math:`\vec{\boldsymbol{y}}_{jn}` follows a Laplace distribution.

    .. math::
        p(\vec{\boldsymbol{y}}_{jn})
        \propto\frac{1}{\alpha_{jn}^{I}}
        \exp\left(\frac{\|\vec{\boldsymbol{y}}_{jn}\|_{2}}{\alpha_{jn}}\right)

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the gradient descent if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.

    Examples:
        Update demixing filters using Holonomic-type update:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = NaturalGradLaplaceIVA(is_holonomic=True)
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=500)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters using Nonholonomic-type update:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = NaturalGradLaplaceIVA(is_holonomic=False)
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=500)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)
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
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        def contrast_fn(y: np.ndarray) -> np.ndarray:
            r"""Contrast function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray of the shape is (n_sources, n_frames).
            """
            return 2 * np.linalg.norm(y, axis=1)

        def score_fn(y: np.ndarray) -> np.ndarray:
            r"""Score function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray of the shape is (n_sources, n_bins, n_frames).
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
            scale_restoration=scale_restoration,
            record_loss=record_loss,
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

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L} \
            = \frac{2}{J}\sum_{j,n}\|\vec{\boldsymbol{y}}_{jn}\|_{2} \
            - 2\sum_{i}\log|\det\boldsymbol{W}_{i}|.

        Returns:
            Computed loss.
        """
        return super().compute_loss()


class NaturalGradGaussIVA(NaturalGradIVA):
    r"""Independent vector analysis (IVA) using the natural gradient descent \
    on a time-varying Gaussian distribution.

    We assume :math:`\vec{\boldsymbol{y}}_{jn}` follows a time-varying Gaussian distribution.

    .. math::
        p(\vec{\boldsymbol{y}}_{jn})
        \propto\frac{1}{\alpha_{jn}^{I}}
        \exp\left(\frac{\|\vec{\boldsymbol{y}}_{jn}\|_{2}^{2}}{\alpha_{jn}}\right).

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the gradient descent if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.

    Examples:
        Update demixing filters using Holonomic-type update:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = NaturalGradGaussIVA(is_holonomic=True)
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=500)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters using Nonholonomic-type update:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = NaturalGradGaussIVA(is_holonomic=False)
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=500)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[
                Callable[["NaturalGradGaussIVA"], None],
                List[Callable[["NaturalGradGaussIVA"], None]],
            ]
        ] = None,
        is_holonomic: bool = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        def contrast_fn(y: np.ndarray) -> np.ndarray:
            r"""
            Args:
                y (numpy.ndarray):
                    Separated signal with shape of (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray of computed contrast function.
                The shape is (n_sources, n_frames).
            """
            n_bins = self.n_bins
            alpha = self.variance
            norm = np.linalg.norm(y, axis=1)

            return n_bins * np.log(alpha) + (norm**2) / alpha

        def score_fn(y: np.ndarray) -> np.ndarray:
            r"""
            Args:
                y (numpy.ndarray):
                    Norm of separated signal.
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray of computed contrast function.
                The shape is (n_sources, n_frames).
            """
            alpha = self.variance
            return y / alpha[:, np.newaxis, :]

        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            is_holonomic=is_holonomic,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

    def _reset(self, **kwargs) -> None:
        r"""Reset attributes by given keyword arguments.

        We also set variance of Gaussian distribution.

        Args:
            kwargs:
                Keyword arguments to set as attributes of IVA.
        """
        super()._reset(**kwargs)

        n_sources, n_frames = self.n_sources, self.n_frames

        self.variance = np.ones((n_sources, n_frames))

    def update_once(self) -> None:
        r"""Update variance and demixing filters and once."""
        self.update_source_model()

        super().update_once()

    def update_source_model(self) -> None:
        r"""Update variance of Gaussian distribution."""
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        self.variance = np.mean(np.abs(Y) ** 2, axis=1)


class AuxLaplaceIVA(AuxIVA):
    r"""Auxiliary-function-based independent vector analysis (IVA) \
    on a Laplace distribution.

    We assume :math:`\vec{\boldsymbol{y}}_{jn}` follows a Laplace distribution.

    .. math::
        p(\vec{\boldsymbol{y}}_{jn})\propto\exp(\|\vec{\boldsymbol{y}}_{jn}\|_{2})

    Args:
        spatial_algorithm (str):
            Algorithm for demixing filter updates.
            Choose ``IP``, ``IP1``, ``IP2``, ``ISS``, ``ISS1``, or ``ISS2``.
            Default: ``IP``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
        pair_selector (callable, optional):
            Selector to choose updaing pair in ``IP2`` and ``ISS2``.
            If ``None`` is given, ``sequential_pair_selector`` is used.
            Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the demixing filter update if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.

    Examples:
        Update demixing filters by IP:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = AuxLaplaceIVA(spatial_algorithm="IP")
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by IP2:

        .. code-block:: python

            >>> from ssspy.utils.select_pair import sequential_pair_selector

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = AuxLaplaceIVA(
            ...     spatial_algorithm="IP2",
            ...     pair_selector=sequential_pair_selector,
            ... )
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by ISS:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = AuxLaplaceIVA(spatial_algorithm="ISS")
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by ISS2:

        .. code-block:: python

            >>> import functools
            >>> from ssspy.utils.select_pair import sequential_pair_selector

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = AuxLaplaceIVA(
            ...     spatial_algorithm="ISS2",
            ...     pair_selector=functools.partial(sequential_pair_selector, step=2),
            ... )
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        spatial_algorithm: str = "IP",
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        pair_selector: Optional[Callable[[int], Iterable[Tuple[int, int]]]] = None,
        callbacks: Optional[
            Union[Callable[["AuxLaplaceIVA"], None], List[Callable[["AuxLaplaceIVA"], None]]]
        ] = None,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
        **kwargs,
    ) -> None:
        def contrast_fn(y) -> np.ndarray:
            r"""Contrast function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray of the shape is (n_sources, n_frames).
            """
            return 2 * np.linalg.norm(y, axis=1)

        def d_contrast_fn(y) -> np.ndarray:
            r"""Derivative of contrast function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray of the shape is (n_sources, n_frames).
            """
            return 2 * np.ones_like(y)

        super().__init__(
            spatial_algorithm=spatial_algorithm,
            contrast_fn=contrast_fn,
            d_contrast_fn=d_contrast_fn,
            flooring_fn=flooring_fn,
            pair_selector=pair_selector,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
            **kwargs,
        )


class AuxGaussIVA(AuxIVA):
    r"""Auxiliary-function-based independent vector analysis (IVA) \
    on a time-varying Gaussian distribution [#ono2012auxiliary]_.

    We assume :math:`\vec{\boldsymbol{y}}_{jn}` follows a time-varying Gaussian distribution.

    .. math::
        p(\vec{\boldsymbol{y}}_{jn})
        \propto\frac{1}{\alpha_{jn}^{I}}
        \exp\left(\frac{\|\vec{\boldsymbol{y}}_{jn}\|_{2}^{2}}{\alpha_{jn}}\right).

    Args:
        spatial_algorithm (str):
            Algorithm for demixing filter updates.
            Choose ``IP``, ``IP1``, ``IP2``, ``ISS``, ``ISS1``, or ``ISS2``.
            Default: ``IP``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
        pair_selector (callable, optional):
            Selector to choose updaing pair in ``IP2`` and ``ISS2``.
            If ``None`` is given, ``sequential_pair_selector`` is used.
            Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back``
            or ``minimal_distortion_principle``. Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the demixing filter update if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back and minimal distortion principle. Default: ``0``.

    Examples:
        Update demixing filters by IP:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = AuxGaussIVA(spatial_algorithm="IP")
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by IP2:

        .. code-block:: python

            >>> from ssspy.utils.select_pair import sequential_pair_selector

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = AuxGaussIVA(
            ...     spatial_algorithm="IP2",
            ...     pair_selector=sequential_pair_selector,
            ... )
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by ISS:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = AuxGaussIVA(spatial_algorithm="ISS")
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by ISS2:

        .. code-block:: python

            >>> import functools
            >>> from ssspy.utils.select_pair import sequential_pair_selector

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> iva = AuxGaussIVA(
            ...     spatial_algorithm="ISS2",
            ...     pair_selector=functools.partial(sequential_pair_selector, step=2),
            ... )
            >>> spectrogram_est = iva(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

    .. [#ono2012auxiliary]
        N. Ono,
        "Auxiliary-function-based independent vector analysis with power of \
        vector-norm type weighting functions,"
        in *Proc. APSIPA ASC*, 2012, pp. 1-4.
    """

    def __init__(
        self,
        spatial_algorithm: str = "IP",
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        pair_selector: Optional[Callable[[int], Iterable[Tuple[int, int]]]] = None,
        callbacks: Optional[
            Union[Callable[["AuxGaussIVA"], None], List[Callable[["AuxGaussIVA"], None]]]
        ] = None,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
        **kwargs,
    ) -> None:
        def contrast_fn(y: np.ndarray) -> np.ndarray:
            r"""
            Args:
                y (numpy.ndarray):
                    Separated signal with shape of (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray:
                    Computed contrast function.
                    The shape is (n_sources, n_frames).
            """
            n_bins = self.n_bins
            alpha = self.variance
            norm = np.linalg.norm(y, axis=1)

            return n_bins * np.log(alpha) + (norm**2) / alpha

        def d_contrast_fn(y: np.ndarray, variance: np.ndarray = None) -> np.ndarray:
            r"""
            Args:
                y (numpy.ndarray):
                    Norm of separated signal.
                    The shape is (n_sources, n_frames).

            Returns:
                numpy.ndarray of computed contrast function.
                The shape is (n_sources, n_frames).
            """
            if variance is None:
                alpha = self.variance
            else:
                alpha = variance

            return 2 * y / alpha

        super().__init__(
            spatial_algorithm=spatial_algorithm,
            contrast_fn=contrast_fn,
            d_contrast_fn=d_contrast_fn,
            flooring_fn=flooring_fn,
            pair_selector=pair_selector,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
            **kwargs,
        )

    def _reset(self, **kwargs) -> None:
        r"""Reset attributes by given keyword arguments.

        We also set variance of Gaussian distribution.

        Args:
            kwargs:
                Keyword arguments to set as attributes of IVA.
        """
        super()._reset(**kwargs)

        n_sources, n_frames = self.n_sources, self.n_frames

        self.variance = np.ones((n_sources, n_frames))

    def update_once(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update variance and demixing filters and once.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """
        self.update_source_model()

        super().update_once(flooring_fn=flooring_fn)

    def update_once_ip2(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update demixing filters once using pairwise iterative projection.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        For :math:`n_{1}` and :math:`n_{2}` (:math:`n_{1}\neq n_{2}`),
        compute auxiliary variables:

        .. math::
            \bar{r}_{jn_{1}}
            &\leftarrow\|\vec{\boldsymbol{y}}_{jn_{1}}\|_{2} \\
            \bar{r}_{jn_{2}}
            &\leftarrow\|\vec{\boldsymbol{y}}_{jn_{2}}\|_{2}

        Then, for :math:`n=n_{1},n_{2}`, compute weighted covariance matrix as follows:

        .. math::
            \boldsymbol{U}_{in_{1}}
            &= \frac{1}{J}\sum_{j}
            \varphi(\bar{r}_{jn_{1}})\boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}, \\
            \boldsymbol{U}_{in_{2}}
            &= \frac{1}{J}\sum_{j}
            \varphi(\bar{r}_{jn_{2}})\boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}},

        where

        .. math::
            \varphi(\bar{r}_{jn})
            = \frac{G'_{\mathbb{R}}(\bar{r}_{jn})}{2\bar{r}_{jn}}.

        Using :math:`\boldsymbol{U}_{in_{1}}` and
        :math:`\boldsymbol{U}_{in_{2}}`, we compute generalized eigenvectors.

        .. math::
            \left({\boldsymbol{P}_{in_{1}}^{(n_{1},n_{2})}}^{\mathsf{H}}\boldsymbol{U}_{in_{1}}
            \boldsymbol{P}_{in_{1}}^{(n_{1},n_{2})}\right)\boldsymbol{h}_{i}
            = \lambda_{i}
            \left({\boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}}^{\mathsf{H}}\boldsymbol{U}_{in_{2}}
            \boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}\right)\boldsymbol{h}_{i},

        where

        .. math::
            \boldsymbol{P}_{in_{1}}^{(n_{1},n_{2})}
            &= (\boldsymbol{W}_{i}\boldsymbol{U}_{in_{1}})^{-1}
            (
            \begin{array}{cc}
                \boldsymbol{e}_{n_{1}} & \boldsymbol{e}_{n_{2}}
            \end{array}
            ), \\
            \boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}
            &= (\boldsymbol{W}_{i}\boldsymbol{U}_{in_{2}})^{-1}
            (
            \begin{array}{cc}
                \boldsymbol{e}_{n_{1}} & \boldsymbol{e}_{n_{2}}
            \end{array}
            ).

        After that, we standardize two eigenvectors :math:`\boldsymbol{h}_{in_{1}}`
        and :math:`\boldsymbol{h}_{in_{2}}`.

        .. math::
            \boldsymbol{h}_{in_{1}}
            &\leftarrow\frac{\boldsymbol{h}_{in_{1}}}
            {\sqrt{\boldsymbol{h}_{in_{1}}^{\mathsf{H}}
            \left({\boldsymbol{P}_{in_{1}}^{(n_{1},n_{2})}}^{\mathsf{H}}\boldsymbol{U}_{in_{1}}
            \boldsymbol{P}_{in_{1}}^{(n_{1},n_{2})}\right)
            \boldsymbol{h}_{in_{1}}}}, \\
            \boldsymbol{h}_{in_{2}}
            &\leftarrow\frac{\boldsymbol{h}_{in_{2}}}
            {\sqrt{\boldsymbol{h}_{in_{2}}^{\mathsf{H}}
            \left({\boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}}^{\mathsf{H}}\boldsymbol{U}_{in_{2}}
            \boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}\right)
            \boldsymbol{h}_{in_{2}}}}.

        Then, update :math:`\boldsymbol{w}_{in_{1}}` and :math:`\boldsymbol{w}_{in_{2}}`
        simultaneously.

        .. math::
            \boldsymbol{w}_{in_{1}}
            &\leftarrow \boldsymbol{P}_{in_{1}}^{(n_{1},n_{2})}\boldsymbol{h}_{in_{1}} \\
            \boldsymbol{w}_{in_{2}}
            &\leftarrow \boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}\boldsymbol{h}_{in_{2}}.

        At each iteration, we update pairs of :math:`n_{1}` and :math:`n_{1}`
        for :math:`n_{1}\neq n_{2}`.
        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        n_sources = self.n_sources

        X, W = self.input, self.demix_filter
        R = self.variance

        XX_Hermite = X[:, np.newaxis, :, :] * X[np.newaxis, :, :, :].conj()
        XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)

        for m, n in self.pair_selector(n_sources):
            W_mn = W[:, (m, n), :]
            Y_mn = self.separate(X, demix_filter=W_mn)
            R_mn = R[(m, n), :]

            norm = np.linalg.norm(Y_mn, axis=1)
            weight_mn = self.d_contrast_fn(norm, variance=R_mn) / flooring_fn(2 * norm)
            GXX_mn = weight_mn[:, np.newaxis, np.newaxis, :] * XX_Hermite[:, np.newaxis, :, :, :]
            U_mn = np.mean(GXX_mn, axis=-1)

            W[:, (m, n), :] = update_by_ip2_one_pair(
                W,
                U_mn,
                pair=(m, n),
                flooring_fn=flooring_fn,
            )

        self.demix_filter = W

    def update_source_model(self) -> None:
        r"""Update variance of Gaussian distribution."""
        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        self.variance = np.mean(np.abs(Y) ** 2, axis=1)
