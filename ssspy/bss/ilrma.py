import functools
import warnings
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

from ..algorithm import (
    MINIMAL_DISTORTION_PRINCIPLE_KEYWORDS,
    PROJECTION_BACK_KEYWORDS,
    minimal_distortion_principle,
    projection_back,
)
from ..special.flooring import identity, max_flooring
from ..utils.flooring import choose_flooring_fn
from ..utils.select_pair import sequential_pair_selector
from ._update_spatial_model import (
    update_by_ip1,
    update_by_ip2,
    update_by_ipa,
    update_by_iss1,
    update_by_iss2,
)
from .base import IterativeMethodBase

__all__ = ["GaussILRMA", "TILRMA", "GGDILRMA"]

spatial_algorithms = ["IP", "IP1", "IP2", "ISS", "ISS1", "ISS2", "IPA"]
source_algorithms = ["MM", "ME"]
EPS = 1e-10


class ILRMABase(IterativeMethodBase):
    r"""Base class of independent low-rank matrix analysis (ILRMA).

    Args:
        n_basis (int):
            Number of NMF bases.
        partitioning (bool):
            Whether to use partioning function. Default: ``False``.
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
            estimated spectrograms. You can also specify ``projection_back`` explicitly.
            Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.
        rng (numpy.random.Generator, optioinal):
            Random number generator. This is mainly used to randomly initialize NMF.
            If ``None`` is given, ``np.random.default_rng()`` is used.
            Default: ``None``.
    """

    def __init__(
        self,
        n_basis: int,
        partitioning: bool = False,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["ILRMABase"], None], List[Callable[["ILRMABase"], None]]]
        ] = None,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(callbacks=callbacks, record_loss=record_loss)

        self.n_basis = n_basis
        self.partitioning = partitioning

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

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

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

        self._reset(flooring_fn=self.flooring_fn, **kwargs)

        super().__call__(n_iter=n_iter, initial_call=initial_call)

        if self.scale_restoration:
            self.restore_scale()

        self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "ILRMA("
        s += "n_basis={n_basis}"
        s += ", partitioning={partitioning}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
        **kwargs,
    ) -> None:
        r"""Reset attributes by given keyword arguments.

        We also set variance of Gaussian distribution.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.
            kwargs:
                Keyword arguments to set as attributes of ILRMA.
        """
        assert self.input is not None, "Specify data!"

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

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

        self._init_nmf(flooring_fn=flooring_fn, rng=self.rng)

    def _init_nmf(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        r"""Initialize NMF.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.
            rng (numpy.random.Generator, optional):
                Random number generator. If ``None`` is given,
                ``np.random.default_rng()`` is used.
                Default: ``None``.
        """
        n_basis = self.n_basis
        n_sources = self.n_sources
        n_bins, n_frames = self.n_bins, self.n_frames

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if rng is None:
            rng = np.random.default_rng()

        if self.partitioning:
            if not hasattr(self, "latent"):
                Z = rng.random((n_sources, n_basis))
                Z = Z / Z.sum(axis=0)
                Z = flooring_fn(Z)
            else:
                # To avoid overwriting.
                Z = self.latent.copy()

            if not hasattr(self, "basis"):
                T = rng.random((n_bins, n_basis))
                T = flooring_fn(T)
            else:
                # To avoid overwriting.
                T = self.basis.copy()

            if not hasattr(self, "activation"):
                V = rng.random((n_basis, n_frames))
                V = flooring_fn(V)
            else:
                # To avoid overwriting.
                V = self.activation.copy()

            self.latent = Z
            self.basis, self.activation = T, V
        else:
            if not hasattr(self, "basis"):
                T = rng.random((n_sources, n_bins, n_basis))
                T = flooring_fn(T)
            else:
                # To avoid overwriting.
                T = self.basis.copy()

            if not hasattr(self, "activation"):
                V = rng.random((n_sources, n_basis, n_frames))
                V = flooring_fn(V)
            else:
                # To avoid overwriting.
                V = self.activation.copy()

            self.basis, self.activation = T, V

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

    def reconstruct_nmf(
        self, basis: np.ndarray, activation: np.ndarray, latent: Optional[np.ndarray] = None
    ) -> np.ndarray:
        r"""Reconstruct NMF.

        Args:
            basis (numpy.ndarray):
                Basis matrix.
                The shape is (n_sources, n_basis, n_frames) if latent is given.
                Otherwise, (n_basis, n_frames).
            activation (numpy.ndarray):
                Activation matrix.
                The shape is (n_sources, n_bins, n_basis) if latent is given.
                Otherwise, (n_bins, n_basis).
            latent (numpy.ndarray, optional):
                Latent variable that determines number of bases per source.

        Returns:
            numpy.ndarray of theconstructed NMF.
            The shape is (n_sources, n_bins, n_frames).
        """
        if latent is None:
            T, V = basis, activation
            R = T @ V
        else:
            Z = latent
            T, V = basis, activation
            TV = T[:, :, np.newaxis] * V[np.newaxis, :, :]
            R = np.sum(Z[:, np.newaxis, :, np.newaxis] * TV[np.newaxis, :, :, :], axis=2)

        return R

    def update_once(self) -> None:
        r"""Update demixing filters once."""
        raise NotImplementedError("Implement 'update_once' method.")

    def normalize(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Normalize demixing filters and NMF parameters.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """
        normalization = self.normalization
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        assert normalization, "Set normalization."

        if type(normalization) is bool:
            # when normalization is True
            normalization = "power"

        if normalization == "power":
            self.normalize_by_power(flooring_fn=flooring_fn)
        elif normalization == "projection_back":
            self.normalize_by_projection_back()
        else:
            raise NotImplementedError("Normalization {} is not implemented.".format(normalization))

    def normalize_by_power(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Normalize demixing filters and NMF parameters by power.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Demixing filters are normalized by

        .. math::
            \boldsymbol{w}_{in}
            \leftarrow\frac{\boldsymbol{w}_{in}}{\psi_{in}},

        where

        .. math::
            \psi_{in}
            = \sqrt{\frac{1}{IJ}|\boldsymbol{w}_{in}^{\mathsf{H}}
            \boldsymbol{x}_{ij}|^{2}}.

        For NMF parameters,

        .. math::
            t_{ik}
            &\leftarrow t_{ik}\sum_{n}\frac{z_{nk}}{\psi_{in}^{p}}, \\
            z_{nk}
            &\leftarrow \frac{\frac{z_{nk}}{\psi_{in}^{p}}}
            {\sum_{n'}\frac{z_{n'k}}{\psi_{in'}^{p}}},

        if ``self.partitioning=True``. Otherwise,

        .. math::
            t_{ikn}
            \leftarrow\frac{t_{ikn}}{\psi_{in}^{p}}.
        """
        p = self.domain

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Y2 = np.mean(np.abs(Y) ** 2, axis=(-2, -1))
        psi = np.sqrt(Y2)
        psi = flooring_fn(psi)

        if self.partitioning:
            Z, T = self.latent, self.basis

            Z_psi = Z / (psi[:, np.newaxis] ** p)
            scale = np.sum(Z_psi, axis=0)
            T = T * scale[np.newaxis, :]
            Z = Z_psi / scale

            self.latent, self.basis = Z, T
        else:
            T = self.basis

            T = T / (psi[:, np.newaxis, np.newaxis] ** p)

            self.basis = T

        if self.demix_filter is None:
            Y = Y / psi[:, np.newaxis, np.newaxis]
            self.output = Y
        else:
            W = self.demix_filter
            W = W / psi[np.newaxis, :, np.newaxis]
            self.demix_filter = W

    def normalize_by_projection_back(self) -> None:
        r"""Normalize demixing filters and NMF parameters by projection back.

        Demixing filters are normalized by

        .. math::
            \boldsymbol{w}_{in}
            \leftarrow\frac{\boldsymbol{w}_{in}}{\psi_{in}},

        where

        .. math::
            \boldsymbol{\psi}_{i}
            = \boldsymbol{W}_{i}^{-1}\boldsymbol{e}_{m_{\mathrm{ref}}}.

        For NMF parameters,

        .. math::
            t_{ikn}
            \leftarrow\frac{t_{ikn}}{\psi_{in}^{p}}.
        """
        p = self.domain
        reference_id = self.reference_id

        X = self.input

        if reference_id is None:
            warnings.warn(
                "channel 0 is used for reference_id \
                    of projection-back-based normalization.",
                UserWarning,
            )
            reference_id = 0

        if self.partitioning:
            raise NotImplementedError(
                "Projection-back-based normalization is not applicable with partitioning function."
            )
        else:
            T = self.basis

            if self.demix_filter is None:
                Y = self.output

                Y = Y.transpose(1, 0, 2)  # (n_bins, n_sources, n_frames)
                X = X.transpose(1, 0, 2)  # (n_bins, n_channels, n_frames)
                Y_Hermite = Y.transpose(0, 2, 1).conj()  # (n_bins, n_frames, n_sources)
                XY_Hermite = X @ Y_Hermite  # (n_bins, n_channels, n_sources)
                YY_Hermite = Y @ Y_Hermite  # (n_bins, n_sources, n_sources)
                scale = XY_Hermite @ np.linalg.inv(YY_Hermite)  # (n_bins, n_channels, n_sources)
                scale = scale[..., reference_id, :]  # (n_bins, n_sources)
                Y_scaled = Y * scale[..., np.newaxis]  # (n_bins, n_sources, n_frames)
                Y = Y_scaled.swapaxes(-3, -2)  # (n_sources, n_bins, n_frames)

                self.output = Y
            else:
                W = self.demix_filter

                scale = np.linalg.inv(W)
                scale = scale[:, reference_id, :]
                W = W * scale[:, :, np.newaxis]

                self.demix_filter = W

            scale = scale.transpose(1, 0)
            scale = np.abs(scale) ** p
            T = T * scale[:, :, np.newaxis]

            self.basis = T

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        Returns:
            Computed loss.
        """
        raise NotImplementedError("Implement 'compute_loss' method.")

    def compute_logdet(self, demix_filter: np.ndarray) -> np.ndarray:
        r"""Compute log-determinant of demixing filter

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

        If ``self.scale_restoration="projection_back``, we use projection back technique.
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


class GaussILRMA(ILRMABase):
    r"""Independent low-rank matrix analysis (ILRMA) [#kitamura2016determined]_ \
    on Gaussian distribution.

    We assume :math:`y_{ijn}` follows a Gaussian distribution.

    .. math::
        p(y_{ijn})
        = \frac{1}{\pi r_{ijn}}\exp\left(-\frac{|y_{ijn}|^{2}}{r_{ijn}}\right),

    where

    .. math::
        r_{ijn}
        = \left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}},

    if ``partitioning=True``. Otherwise,

    .. math::
        r_{ijn}
        = \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}.

    Args:
        n_basis (int):
            Number of NMF bases.
        spatial_algorithm (str):
            Algorithm for demixing filter updates.
            Choose ``IP``, ``IP1``, ``IP2``, ``ISS``, ``ISS1``, or ``ISS2``.
            Default: ``IP``.
        source_algorithm (str):
            Algorithm for source model updates.
            Choose ``MM`` or ``ME``. Default: ``MM``.
        domain (float):
            Domain parameter. Default: ``2``.
        partitioning (bool):
            Whether to use partioning function. Default: ``False``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        pair_selector (callable, optional):
            Selector to choose updaing pair in ``IP2`` and ``ISS2``.
            If ``None`` is given, ``sequential_pair_selector`` is used.
            Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        normalization (bool or str, optional):
            Normalization of demixing filters and NMF parameters.
            Choose ``power`` or ``projection_back``.
            Default: ``power``.
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
        rng (numpy.random.Generator, optioinal):
            Random number generator. This is mainly used to randomly initialize NMF.
            If ``None`` is given, ``np.random.default_rng()`` is used.
            Default: ``None``.

    Examples:
        Update demixing filters by IP:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> ilrma = GaussILRMA(
            ...     n_basis=2,
            ...     spatial_algorithm="IP",
            ...     rng=np.random.default_rng(42),
            ... )
            >>> spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by IP2:

        .. code-block:: python

            >>> from ssspy.utils.select_pair import sequential_pair_selector

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> ilrma = GaussILRMA(
            ...     n_basis=2,
            ...     spatial_algorithm="IP2",
            ...     pair_selector=sequential_pair_selector,
            ...     rng=np.random.default_rng(42),
            ... )
            >>> spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by ISS:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> ilrma = GaussILRMA(
            ...     n_basis=2,
            ...     spatial_algorithm="ISS",
            ...     rng=np.random.default_rng(42),
            ... )
            >>> spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by ISS2:

        .. code-block:: python

            >>> import functools
            >>> from ssspy.utils.select_pair import sequential_pair_selector

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> ilrma = GaussILRMA(
            ...     n_basis=2,
            ...     spatial_algorithm="ISS2",
            ...     pair_selector=functools.partial(sequential_pair_selector, step=2),
            ...     rng=np.random.default_rng(42),
            ... )
            >>> spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

    Update demixing filters by IPA:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> ilrma = GaussILRMA(
            ...     n_basis=2,
            ...     spatial_algorithm="IPA",
            ...     rng=np.random.default_rng(42),
            ... )
            >>> spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

    .. [#kitamura2016determined] D. Kitamura, N. Ono, H. Sawada, H. Kameoka, and H. Saruwatari, \
        "Determined blind source separation unifying independent vector analysis and \
        nonnegative matrix factorization," \
        *IEEE/ACM Trans. ASLP*, vol. 24, no. 9, pp. 1626-1641, 2016.
    """

    _ipa_default_kwargs = {"lqpqm_normalization": True, "newton_iter": 1}
    _default_kwargs = _ipa_default_kwargs

    def __init__(
        self,
        n_basis: int,
        spatial_algorithm: str = "IP",
        source_algorithm: str = "MM",
        domain: float = 2,
        partitioning: bool = False,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        pair_selector: Optional[Callable[[int], Iterable[Tuple[int, int]]]] = None,
        callbacks: Optional[
            Union[Callable[["GaussILRMA"], None], List[Callable[["GaussILRMA"], None]]]
        ] = None,
        normalization: Optional[Union[bool, str]] = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_basis=n_basis,
            partitioning=partitioning,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
            rng=rng,
        )

        assert spatial_algorithm in spatial_algorithms, "Not support {}.".format(spatial_algorithm)
        assert source_algorithm in source_algorithms, "Not support {}.".format(source_algorithm)
        assert 0 < domain <= 2, "domain parameter should be chosen from [0, 2]."

        if source_algorithm == "ME":
            assert domain == 2, "domain parameter should be 2 when you specify ME algorithm."

        self.spatial_algorithm = spatial_algorithm
        self.source_algorithm = source_algorithm
        self.domain = domain
        self.normalization = normalization

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

        self._reset(flooring_fn=self.flooring_fn, **kwargs)

        # Call __call__ of ILRMABase's parent, i.e. __call__ of IterativeMethodBase
        super(ILRMABase, self).__call__(n_iter=n_iter, initial_call=initial_call)

        if self.scale_restoration:
            self.restore_scale()

        if self.demix_filter is None:
            pass
        else:
            self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "GaussILRMA("
        s += "n_basis={n_basis}"
        s += ", spatial_algorithm={spatial_algorithm}"
        s += ", source_algorithm={source_algorithm}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalization={normalization}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
        **kwargs,
    ) -> None:
        r"""Reset attributes by given keyword arguments.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.
            kwargs:
                Keyword arguments to set as attributes of ILRMA.
        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        super()._reset(flooring_fn=flooring_fn, **kwargs)

        if self.spatial_algorithm in ["ISS", "ISS1", "ISS2", "IPA"]:
            self.demix_filter = None

    def update_once(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF parameters and demixing filters once.

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

        self.update_source_model(flooring_fn=flooring_fn)
        self.update_spatial_model(flooring_fn=flooring_fn)

        if self.normalization:
            self.normalize(flooring_fn=flooring_fn)

    def update_source_model(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF bases, activations, and latent variables.

        - If ``source_algorithm`` is ``MM``, ``update_source_model_mm`` is called.
        - If ``source_algorithm`` is ``ME``, ``update_source_model_me`` is called.

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

        if self.source_algorithm == "MM":
            self.update_source_model_mm(flooring_fn=flooring_fn)
        elif self.source_algorithm == "ME":
            self.update_source_model_me(flooring_fn=flooring_fn)
        else:
            raise ValueError(
                "{}-algorithm-based source model updates are not supported.".format(
                    self.source_algorithm
                )
            )

    def update_source_model_mm(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF bases, activations, and latent variables by MM algorithm.

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

        if self.partitioning:
            self.update_latent_mm()

        self.update_basis_mm(flooring_fn=flooring_fn)
        self.update_activation_mm(flooring_fn=flooring_fn)

    def update_source_model_me(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF bases, activations, and latent variables by ME algorithm.

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

        if self.domain != 2:
            raise ValueError("Domain parameter is expected 2, but given {}.".format(self.domain))

        if self.partitioning:
            self.update_latent_me()

        self.update_basis_me(flooring_fn=flooring_fn)
        self.update_activation_me(flooring_fn=flooring_fn)

    def update_latent_mm(self) -> None:
        r"""Update latent variables in NMF by MM algorithm.

        Update :math:`z_{nk}` as follows:

        .. math::
            z_{nk}
            &\leftarrow\left[\frac{\displaystyle\sum_{i,j}\frac{t_{ik}v_{kj}}
            {(\sum_{k'}z_{nk'}t_{ik'}v_{k'j})^{\frac{p+2}{p}}}
            |y_{ijn}|^{2}}{\displaystyle\sum_{i,j}\dfrac{t_{ik}v_{kj}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            \right]^{\frac{p}{p+2}}z_{nk} \\
            z_{nk}
            &\leftarrow\frac{z_{nk}}{\sum_{n'}z_{n'k}}.
        """
        p = self.domain

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Y2 = np.abs(Y) ** 2
        p2_p = (p + 2) / p
        p_p2 = p / (p + 2)

        Z = self.latent
        T, V = self.basis, self.activation

        TV = T[:, :, np.newaxis] * V[np.newaxis, :, :]
        ZTV = self.reconstruct_nmf(T, V, latent=Z)

        ZTVp2p = ZTV**p2_p
        TV_ZTVp2p = TV[np.newaxis, :, :, :] / ZTVp2p[:, :, np.newaxis, :]
        num = np.sum(TV_ZTVp2p * Y2[:, :, np.newaxis, :], axis=(1, 3))

        TV_ZTV = TV[np.newaxis, :, :, :] / ZTV[:, :, np.newaxis, :]
        denom = np.sum(TV_ZTV, axis=(1, 3))

        Z = ((num / denom) ** p_p2) * Z
        Z = Z / Z.sum(axis=0)

        self.latent = Z

    def update_basis_mm(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF bases.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Update :math:`t_{ikn}` as follows:

        .. math::
            t_{ik}
            \leftarrow\left[
            \frac{\displaystyle\sum_{j,n}\frac{z_{nk}v_{kj}}
            {(\sum_{k'}z_{nk'}t_{ik'}v_{k'j})^{\frac{p+2}{p}}}
            |y_{ijn}|^{2}}{\displaystyle\sum_{j,n}
            \dfrac{z_{nk}v_{kj}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            \right]^{\frac{p}{p+2}}t_{ik},

        if ``partitioning=True``. Otherwise

        .. math::
            t_{ikn}
            \leftarrow \left[\frac{\displaystyle\sum_{j}
            \dfrac{v_{kjn}}{(\sum_{k'}t_{ik'n}v_{k'jn})^{\frac{p+2}{p}}}|y_{ijn}|^{2}}
            {\displaystyle\sum_{j}\frac{v_{kjn}}{\sum_{k'}t_{ik'n}v_{k'jn}}}\right]
            ^{\frac{p}{p+2}}t_{ikn}.
        """
        p = self.domain

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Y2 = np.abs(Y) ** 2
        p2_p = (p + 2) / p
        p_p2 = p / (p + 2)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZV = Z[:, :, np.newaxis] * V[np.newaxis, :, :]
            ZTV = self.reconstruct_nmf(T, V, latent=Z)

            ZTVp2p = ZTV**p2_p
            ZV_ZTVp2p = ZV[:, np.newaxis, :, :] / ZTVp2p[:, :, np.newaxis, :]
            num = np.sum(ZV_ZTVp2p * Y2[:, :, np.newaxis, :], axis=(0, 3))

            ZV_ZTV = ZV[:, np.newaxis, :, :] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(ZV_ZTV, axis=(0, 3))
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)

            TVp2p = TV**p2_p
            V_TVp2p = V[:, np.newaxis, :, :] / TVp2p[:, :, np.newaxis, :]
            num = np.sum(V_TVp2p * Y2[:, :, np.newaxis, :], axis=3)

            V_TV = V[:, np.newaxis, :, :] / TV[:, :, np.newaxis, :]
            denom = np.sum(V_TV, axis=3)

        T = ((num / denom) ** p_p2) * T
        T = flooring_fn(T)

        self.basis = T

    def update_activation_mm(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF activations by MM algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Update :math:`v_{kjn}` as follows:

        .. math::
            v_{kj}
            \leftarrow\left[\frac{\displaystyle\sum_{i,n}\frac{z_{nk}t_{ik}}
            {(\sum_{k'}z_{nk'}t_{ik'}v_{k'j})^{\frac{p+2}{p}}}
            |y_{ijn}|^{2}}{\displaystyle\sum_{i,n}\dfrac{z_{nk}t_{ik}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            \right]^{\frac{p}{p+2}}v_{kj},

        if ``partitioning=True``. Otherwise

        .. math::
            v_{kjn}
            \leftarrow \left[\frac{\displaystyle\sum_{i}
            \dfrac{t_{ikn}}{(\sum_{k'}t_{ik'n}v_{k'jn})^{\frac{p+2}{p}}}|y_{ijn}|^{2}}
            {\displaystyle\sum_{i}\frac{t_{ikn}}{\sum_{k'}t_{ik'n}v_{k'jn}}}
            \right]^{\frac{p}{p+2}}v_{kjn}.
        """
        p = self.domain
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Y2 = np.abs(Y) ** 2
        p2_p = (p + 2) / p
        p_p2 = p / (p + 2)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZT = Z[:, np.newaxis, :] * T[np.newaxis, :, :]
            ZTV = self.reconstruct_nmf(T, V, latent=Z)

            ZTVp2p = ZTV**p2_p
            ZT_ZTVp2p = ZT[:, :, :, np.newaxis] / ZTVp2p[:, :, np.newaxis, :]
            num = np.sum(ZT_ZTVp2p * Y2[:, :, np.newaxis, :], axis=(0, 1))

            ZT_ZTV = ZT[:, :, :, np.newaxis] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(ZT_ZTV, axis=(0, 1))
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)

            TVp2p = TV**p2_p
            T_TVp2p = T[:, :, :, np.newaxis] / TVp2p[:, :, np.newaxis, :]
            num = np.sum(T_TVp2p * Y2[:, :, np.newaxis, :], axis=1)

            T_TV = T[:, :, :, np.newaxis] / TV[:, :, np.newaxis, :]
            denom = np.sum(T_TV, axis=1)

        V = ((num / denom) ** p_p2) * V
        V = flooring_fn(V)

        self.activation = V

    def update_latent_me(self) -> None:
        r"""Update latent variables in NMF by ME algorithm.

        Update :math:`z_{nk}` as follows:

        .. math::
            z_{nk}
            &\leftarrow\left[\frac{\displaystyle\sum_{i,j}\frac{t_{ik}v_{kj}}
            {(\sum_{k'}z_{nk'}t_{ik'}v_{k'j})^{2}}
            |y_{ijn}|^{2}}{\displaystyle\sum_{i,j}\dfrac{t_{ik}v_{kj}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            \right]z_{nk} \\
            z_{nk}
            &\leftarrow\frac{z_{nk}}{\sum_{n'}z_{n'k}}.
        """
        if self.domain != 2:
            raise ValueError("Domain parameter is expected 2, but given {}.".format(self.domain))

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Y2 = np.abs(Y) ** 2

        Z = self.latent
        T, V = self.basis, self.activation

        TV = T[:, :, np.newaxis] * V[np.newaxis, :, :]
        ZTV = self.reconstruct_nmf(T, V, latent=Z)

        ZTV2 = ZTV**2
        TV_ZTV2 = TV[np.newaxis, :, :, :] / ZTV2[:, :, np.newaxis, :]
        num = np.sum(TV_ZTV2 * Y2[:, :, np.newaxis, :], axis=(1, 3))

        TV_ZTV = TV[np.newaxis, :, :, :] / ZTV[:, :, np.newaxis, :]
        denom = np.sum(TV_ZTV, axis=(1, 3))

        Z = (num / denom) * Z
        Z = Z / Z.sum(axis=0)

        self.latent = Z

    def update_basis_me(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF bases by ME algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Update :math:`t_{ikn}` as follows:

        .. math::
            t_{ik}
            \leftarrow\left[
            \frac{\displaystyle\sum_{j,n}\frac{z_{nk}v_{kj}}
            {(\sum_{k'}z_{nk'}t_{ik'}v_{k'j})^{2}}
            |y_{ijn}|^{2}}{\displaystyle\sum_{j,n}
            \dfrac{z_{nk}v_{kj}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            \right]t_{ik},

        if ``partitioning=True``. Otherwise

        .. math::
            t_{ikn}
            \leftarrow\left[\frac{\displaystyle\sum_{j}
            \dfrac{v_{kjn}}{(\sum_{k'}t_{ik'n}v_{k'jn})^{2}}|y_{ijn}|^{2}}
            {\displaystyle\sum_{j}\frac{v_{kjn}}{\sum_{k'}t_{ik'n}v_{k'jn}}}\right]
            t_{ikn}.
        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.domain != 2:
            raise ValueError("Domain parameter is expected 2, but given {}.".format(self.domain))

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Y2 = np.abs(Y) ** 2

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZV = Z[:, :, np.newaxis] * V[np.newaxis, :, :]
            ZTV = self.reconstruct_nmf(T, V, latent=Z)

            ZTV2 = ZTV**2
            ZV_ZTV2 = ZV[:, np.newaxis, :, :] / ZTV2[:, :, np.newaxis, :]
            num = np.sum(ZV_ZTV2 * Y2[:, :, np.newaxis, :], axis=(0, 3))

            ZV_ZTV = ZV[:, np.newaxis, :, :] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(ZV_ZTV, axis=(0, 3))
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)

            TV2 = TV**2
            V_TV2 = V[:, np.newaxis, :, :] / TV2[:, :, np.newaxis, :]
            num = np.sum(V_TV2 * Y2[:, :, np.newaxis, :], axis=3)

            V_TV = V[:, np.newaxis, :, :] / TV[:, :, np.newaxis, :]
            denom = np.sum(V_TV, axis=3)

        T = (num / denom) * T
        T = flooring_fn(T)

        self.basis = T

    def update_activation_me(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF activations by ME algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Update :math:`t_{ikn}` as follows:

        .. math::
            v_{kj}
            \leftarrow\left[\frac{\displaystyle\sum_{i,n}\frac{z_{nk}t_{ik}}
            {(\sum_{k'}z_{nk'}t_{ik'}v_{k'j})^{2}}
            |y_{ijn}|^{2}}{\displaystyle\sum_{i,n}\dfrac{z_{nk}t_{ik}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            \right]v_{kj},

        if ``partitioning=True``. Otherwise

        .. math::
            v_{kjn}
            \leftarrow \left[\frac{\displaystyle\sum_{i}
            \dfrac{t_{ikn}}{(\sum_{k'}t_{ik'n}v_{k'jn})^{2}}|y_{ijn}|^{2}}
            {\displaystyle\sum_{i}\frac{t_{ikn}}{\sum_{k'}t_{ik'n}v_{k'jn}}}
            \right]v_{kjn}.
        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.domain != 2:
            raise ValueError("Domain parameter is expected 2, but given {}.".format(self.domain))

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Y2 = np.abs(Y) ** 2

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZT = Z[:, np.newaxis, :] * T[np.newaxis, :, :]
            ZTV = self.reconstruct_nmf(T, V, latent=Z)

            ZTV2 = ZTV**2
            ZT_ZTV2 = ZT[:, :, :, np.newaxis] / ZTV2[:, :, np.newaxis, :]
            num = np.sum(ZT_ZTV2 * Y2[:, :, np.newaxis, :], axis=(0, 1))

            ZT_ZTV = ZT[:, :, :, np.newaxis] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(ZT_ZTV, axis=(0, 1))
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)

            TV2 = TV**2
            T_TV2 = T[:, :, :, np.newaxis] / TV2[:, :, np.newaxis, :]
            num = np.sum(T_TV2 * Y2[:, :, np.newaxis, :], axis=1)

            T_TV = T[:, :, :, np.newaxis] / TV[:, :, np.newaxis, :]
            denom = np.sum(T_TV, axis=1)

        V = (num / denom) * V
        V = flooring_fn(V)

        self.activation = V

    def update_spatial_model(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update demixing filters once.

        - If ``spatial_algorithm`` is ``IP`` or ``IP1``, ``update_spatial_model_ip1`` is called.
        - If ``spatial_algorithm`` is ``ISS`` or ``ISS1``, ``update_spatial_model_iss1`` is called.
        - If ``spatial_algorithm`` is ``IP2``, ``update_spatial_model_ip2`` is called.
        - If ``spatial_algorithm`` is ``ISS2``, ``update_spatial_model_iss2`` is called.
        - If ``spatial_algorithm`` is ``IPA``, ``update_spatial_model_ipa`` is called.

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
            self.update_spatial_model_ip1(flooring_fn=flooring_fn)
        elif self.spatial_algorithm in ["IP2"]:
            self.update_spatial_model_ip2(flooring_fn=flooring_fn)
        elif self.spatial_algorithm in ["ISS", "ISS1"]:
            self.update_spatial_model_iss1(flooring_fn=flooring_fn)
        elif self.spatial_algorithm in ["ISS2"]:
            self.update_spatial_model_iss2(flooring_fn=flooring_fn)
        elif self.spatial_algorithm in ["IPA"]:
            self.update_spatial_model_ipa(flooring_fn=flooring_fn)
        else:
            raise NotImplementedError("Not support {}.".format(self.spatial_algorithm))

    def update_spatial_model_ip1(
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

        Demixing filters are updated sequentially for :math:`n=1,\ldots,N` as follows:

        .. math::
            \boldsymbol{w}_{in}
            &\leftarrow\left(\boldsymbol{W}_{in}^{\mathsf{H}}\boldsymbol{U}_{in}\right)^{-1} \
            \boldsymbol{e}_{n}, \\
            \boldsymbol{w}_{in}
            &\leftarrow\frac{\boldsymbol{w}_{in}}
            {\sqrt{\boldsymbol{w}_{in}^{\mathsf{H}}\boldsymbol{U}_{in}\boldsymbol{w}_{in}}},

        where

        .. math::
            \boldsymbol{U}_{in}
            = \frac{1}{J}\sum_{j}
            \frac{1}{\left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}}}
            \boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}

        if ``partitioning=True``, otherwise

        .. math::
            \boldsymbol{U}_{in}
            = \frac{1}{J}\sum_{j}
            \frac{1}{\left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}}
            \boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}.
        """
        p = self.domain
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        X, W = self.input, self.demix_filter

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            ZTV2p = ZTV ** (2 / p)
            varphi = 1 / ZTV2p
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)
            TV2p = TV ** (2 / p)
            varphi = 1 / TV2p

        XX_Hermite = X[:, np.newaxis, :, :] * X[np.newaxis, :, :, :].conj()
        XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)

        varphi = varphi.transpose(1, 0, 2)
        varphi_XX = varphi[:, :, np.newaxis, np.newaxis, :] * XX_Hermite[:, np.newaxis, :, :, :]
        U = np.mean(varphi_XX, axis=-1)

        self.demix_filter = update_by_ip1(W, U, flooring_fn=flooring_fn)

    def update_spatial_model_ip2(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update demixing filters once using pairwise iterative projection \
        following [#nakashima2021faster]_.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        For :math:`n_{1}` and :math:`n_{2}` (:math:`n_{1}\neq n_{2}`),
        compute weighted covariance matrix as follows:

        .. math::
            \boldsymbol{U}_{in}
            = \frac{1}{J}\sum_{j}
            \frac{1}{r_{ijn}}\boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}},

        :math:`r_{ijn}` is computed by

        .. math::
            r_{ijn}
            = \left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}}

        if ``partitioning=True``.
        Otherwise,

        .. math::
            r_{ijn}
            = \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}.

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
            &\leftarrow \boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}\boldsymbol{h}_{in_{2}}

        At each iteration, we update pairs of :math:`n_{1}` and :math:`n_{1}`
        for :math:`n_{1}\neq n_{2}`.

        .. [#nakashima2021faster] T. Nakashima, R. Scheibler, Y. Wakabayashi, and N. Ono, \
            "Faster independent low-rank matrix analysis with pairwise updates of demixing vectors,"
            in *Proc. EUSIPCO*, 2021, pp. 301-305.
        """
        p = self.domain
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        X, W = self.input, self.demix_filter

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation
            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            R = ZTV ** (2 / p)
        else:
            T, V = self.basis, self.activation
            TV = self.reconstruct_nmf(T, V)
            R = TV ** (2 / p)

        varphi = 1 / R

        XX_Hermite = X[:, np.newaxis, :, :] * X[np.newaxis, :, :, :].conj()
        XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)

        varphi = varphi.transpose(1, 0, 2)
        varphi_XX = varphi[:, :, np.newaxis, np.newaxis, :] * XX_Hermite[:, np.newaxis, :, :, :]
        U = np.mean(varphi_XX, axis=-1)

        self.demix_filter = update_by_ip2(
            W, U, flooring_fn=flooring_fn, pair_selector=self.pair_selector
        )

    def update_spatial_model_iss1(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update estimated spectrograms once using iterative source steering.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Update :math:`y_{ijn}` as follows:

        .. math::
            \boldsymbol{y}_{ij}
            & \leftarrow\boldsymbol{y}_{ij} - \boldsymbol{d}_{in}y_{ijn} \\
            d_{inn'}
            &= \begin{cases}
                \dfrac{\displaystyle\sum_{j}\dfrac{1}{r_{ijn}}
                y_{ijn'}y_{ijn}^{*}}{\displaystyle\sum_{j}\dfrac{1}
                {r_{ijn}}|y_{ijn}|^{2}}
                & (n'\neq n) \\
                1 - \dfrac{1}{\sqrt{\displaystyle\dfrac{1}{J}\sum_{j}\dfrac{1}
                {r_{ijn}}
                |y_{ijn}|^{2}}} & (n'=n)
            \end{cases},

        where

        .. math::
            r_{ijn}
            = \left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}},

        if ``partitioning=True``. Otherwise

        .. math::
            r_{ijn}
            = \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}.
        """
        p = self.domain

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        Y = self.output

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation
            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            R = ZTV ** (2 / p)
        else:
            T, V = self.basis, self.activation
            TV = self.reconstruct_nmf(T, V)
            R = TV ** (2 / p)

        varphi = 1 / R

        self.output = update_by_iss1(Y, varphi, flooring_fn=flooring_fn)

    def update_spatial_model_iss2(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update estimated spectrograms once using pairwise iterative source steering.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Compute :math:`\boldsymbol{G}_{in}^{(n_{1},n_{2})}`
        and :math:`\boldsymbol{f}_{in}^{(n_{1},n_{2})}` for :math:`n_{1}\neq n_{2}`:

        .. math::
            \begin{array}{rclc}
                \boldsymbol{G}_{in}^{(n_{1},n_{2})}
                &=& {\displaystyle\frac{1}{J}\sum_{j}}\dfrac{1}{r_{ijn}}
                \boldsymbol{y}_{ij}^{(n_{1},n_{2})}{\boldsymbol{y}_{ij}^{(n_{1},n_{2})}}^{\mathsf{H}}
                &(n=1,\ldots,N), \\
                \boldsymbol{f}_{in}^{(n_{1},n_{2})}
                &=& {\displaystyle\frac{1}{J}\sum_{j}}
                \dfrac{1}{r_{ijn}}y_{ijn}^{*}\boldsymbol{y}_{ij}^{(n_{1},n_{2})}
                &(n\neq n_{1},n_{2}),
            \end{array}

        where

        .. math::
            r_{ijn}
            = \left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}}

        if ``partitioning=True``.
        Otherwise,

        .. math::
            r_{ijn}
            = \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}.

        Using :math:`\boldsymbol{G}_{in}^{(n_{1},n_{2})}` and
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

        where :math:`\boldsymbol{h}_{in}` (:math:`n=n_{1},n_{2}`) is
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
        """
        p = self.domain
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        Y = self.output

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation
            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            R = ZTV ** (2 / p)
        else:
            T, V = self.basis, self.activation
            TV = self.reconstruct_nmf(T, V)
            R = TV ** (2 / p)

        varphi = 1 / R

        self.output = update_by_iss2(
            Y, varphi, flooring_fn=flooring_fn, pair_selector=self.pair_selector
        )

    def update_spatial_model_ipa(
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

        Compute :math:`r_{ijn}` as follows:

        .. math::
            r_{ijn}
            = \left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}},

        if ``partitioning=True``. Otherwise

        .. math::
            r_{ijn}
            = \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}.

        Then, by defining, :math:`\tilde{\boldsymbol{U}}_{in'}`,
        :math:`\boldsymbol{A}_{in}\in\mathbb{R}^{(N-1)\times(N-1)}`,
        :math:`\boldsymbol{b}_{in}\in\mathbb{C}^{N-1}`,
        :math:`\boldsymbol{C}_{in}\in\mathbb{C}^{(N-1)\times(N-1)}`,
        :math:`\boldsymbol{d}_{in}\in\mathbb{C}^{N-1}`,
        and :math:`z_{in}\in\mathbb{R}_{\geq 0}` as follows:

        .. math::

            \tilde{\boldsymbol{U}}_{in'}
            &= \frac{1}{J}\sum_{j}\frac{1}{r_{ijn'}}
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
            - \boldsymbol{d}_{in}^{\mathsf{H}}\boldsymbol{C}_{in}^{-1}\boldsymbol{d}_{in}.

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

        p = self.domain
        normalization = self.lqpqm_normalization
        max_iter = self.newton_iter

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        Y = self.output

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation
            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            R = ZTV ** (2 / p)
        else:
            T, V = self.basis, self.activation
            TV = self.reconstruct_nmf(T, V)
            R = TV ** (2 / p)

        varphi = 1 / R

        self.output = update_by_ipa(
            Y,
            varphi,
            normalization=normalization,
            flooring_fn=flooring_fn,
            max_iter=max_iter,
        )

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L}
            = \frac{1}{J}\sum_{i,j,n}\left(\frac{|y_{ijn}|^{2}}{r_{ijn}}
            + \log r_{ijn}\right)
            - 2\sum_{i}\log|\det\boldsymbol{W}_{i}|,

        where

        .. math::
            r_{ijn}
            = \left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}},

        if ``partitioning=True``. Otherwise

        .. math::
            r_{ijn}
            = \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}.

        Returns:
            Computed loss.
        """
        p = self.domain

        if self.demix_filter is None:
            X, Y = self.input, self.output
            Y2 = np.abs(Y) ** 2
            X, Y = X.transpose(1, 0, 2), Y.transpose(1, 0, 2)
            X_Hermite = X.transpose(0, 2, 1).conj()
            XX_Hermite = X @ X_Hermite
            W = Y @ X_Hermite @ np.linalg.inv(XX_Hermite)
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
            Y2 = np.abs(Y) ** 2

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation
            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            R = ZTV ** (2 / p)
            loss = Y2 / R + (2 / p) * np.log(ZTV)
        else:
            T, V = self.basis, self.activation
            TV = self.reconstruct_nmf(T, V)
            R = TV ** (2 / p)
            loss = Y2 / R + (2 / p) * np.log(TV)

        logdet = self.compute_logdet(W)  # (n_bins,)

        loss = np.sum(loss.mean(axis=-1), axis=0) - 2 * logdet
        loss = loss.sum(axis=0).item()

        return loss

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


class TILRMA(ILRMABase):
    r"""Independent low-rank matrix analysis (ILRMA) on Student's *t* distribution.

    We assume :math:`y_{ijn}` follows a Student's *t* distribution.

    .. math::
        p(y_{ijn})
        = \frac{1}{\pi r_{ijn}}
        \left(1+\frac{2}{\nu}\frac{|y_{ijn}|^{2}}{r_{ijn}}\right)^{-\frac{2+\nu}{2}},

    where

    .. math::
        r_{ijn}
        = \left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}},

    if ``partitioning=True``. Otherwise,

    .. math::
        r_{ijn}
        = \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}.

    :math:`\nu` is a degree of freedom parameter.

    Args:
        n_basis (int):
            Number of NMF bases.
        dof (float):
            Degree of freedom parameter in student's-t distribution.
        spatial_algorithm (str):
            Algorithm for demixing filter updates.
            Choose ``IP``, ``IP1``, ``IP2``, ``ISS``, ``ISS1``, or ``ISS2``.
            Default: ``IP``.
        source_algorithm (str):
            Algorithm for source model updates.
            Choose ``MM`` or ``ME``. Default: ``MM``.
        domain (float):
            Domain parameter. Default: ``2``.
        partitioning (bool):
            Whether to use partioning function. Default: ``False``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        pair_selector (callable, optional):
            Selector to choose updaing pair in ``IP2`` and ``ISS2``.
            If ``None`` is given, ``sequential_pair_selector`` is used.
            Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        normalization (bool or str, optional):
            Normalization of demixing filters and NMF parameters.
            Choose ``power`` or ``projection_back``.
            Default: ``power``.
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
        rng (numpy.random.Generator, optioinal):
            Random number generator. This is mainly used to randomly initialize NMF.
            If ``None`` is given, ``np.random.default_rng()`` is used.
            Default: ``None``.

    Examples:
        Update demixing filters by IP:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> ilrma = TILRMA(
            ...     n_basis=2,
            ...     dof=1000,
            ...     spatial_algorithm="IP",
            ...     rng=np.random.default_rng(42),
            ... )
            >>> spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by IP2:

        .. code-block:: python

            >>> from ssspy.utils.select_pair import sequential_pair_selector

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> ilrma = TILRMA(
            ...     n_basis=2,
            ...     dof=1000,
            ...     spatial_algorithm="IP2",
            ...     pair_selector=sequential_pair_selector,
            ...     rng=np.random.default_rng(42),
            ... )
            >>> spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by ISS:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> ilrma = TILRMA(
            ...     n_basis=2,
            ...     dof=1000,
            ...     spatial_algorithm="ISS",
            ...     rng=np.random.default_rng(42),
            ... )
            >>> spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by ISS2:

        .. code-block:: python

            >>> import functools
            >>> from ssspy.utils.select_pair import sequential_pair_selector

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> ilrma = TILRMA(
            ...     n_basis=2,
            ...     dof=1000,
            ...     spatial_algorithm="ISS2",
            ...     pair_selector=functools.partial(sequential_pair_selector, step=2),
            ...     rng=np.random.default_rng(42),
            ... )
            >>> spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        n_basis: int,
        dof: float,
        spatial_algorithm: str = "IP",
        source_algorithm: str = "MM",
        domain: float = 2,
        partitioning: bool = False,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        pair_selector: Optional[Callable[[int], Iterable[Tuple[int, int]]]] = None,
        callbacks: Optional[
            Union[Callable[["TILRMA"], None], List[Callable[["TILRMA"], None]]]
        ] = None,
        normalization: Optional[Union[bool, str]] = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            n_basis=n_basis,
            partitioning=partitioning,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
            rng=rng,
        )

        assert spatial_algorithm in spatial_algorithms, "Not support {}.".format(spatial_algorithms)
        assert source_algorithm in source_algorithms, "Not support {}.".format(source_algorithm)
        assert 0 < domain <= 2, "domain parameter should be chosen from [0, 2]."

        if spatial_algorithm == "IPA":
            raise ValueError("IPA is not supported for t-ILRMA.")

        if source_algorithm == "ME":
            assert domain == 2, "domain parameter should be 2 when you specify ME algorithm."

        self.dof = dof
        self.spatial_algorithm = spatial_algorithm
        self.source_algorithm = source_algorithm
        self.domain = domain
        self.normalization = normalization

        if pair_selector is None:
            if spatial_algorithm in ["IP2", "ISS2"]:
                self.pair_selector = sequential_pair_selector
        else:
            self.pair_selector = pair_selector

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

        self._reset(flooring_fn=self.flooring_fn, **kwargs)

        # Call __call__ of ILRMABase's parent, i.e. __call__ of IterativeMethodBase
        super(ILRMABase, self).__call__(n_iter=n_iter, initial_call=initial_call)

        if self.scale_restoration:
            self.restore_scale()

        if self.demix_filter is None:
            pass
        else:
            self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "TILRMA("
        s += "n_basis={n_basis}"
        s += ", dof={dof}"
        s += ", spatial_algorithm={spatial_algorithm}"
        s += ", source_algorithm={source_algorithm}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalization={normalization}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
        **kwargs,
    ) -> None:
        r"""Reset attributes by given keyword arguments.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.
            kwargs:
                Keyword arguments to set as attributes of ILRMA.
        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        super()._reset(flooring_fn=flooring_fn, **kwargs)

        if self.spatial_algorithm in ["ISS", "ISS1", "ISS2"]:
            self.demix_filter = None

    def update_once(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF parameters and demixing filters once.

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

        self.update_source_model(flooring_fn=flooring_fn)
        self.update_spatial_model(flooring_fn=flooring_fn)

        if self.normalization:
            self.normalize(flooring_fn=flooring_fn)

    def update_source_model(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF bases, activations, and latent variables.

        - If ``source_algorithm`` is ``MM``, ``update_source_model_mm`` is called.
        - If ``source_algorithm`` is ``ME``, ``update_source_model_me`` is called.

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

        if self.source_algorithm == "MM":
            self.update_source_model_mm(flooring_fn=flooring_fn)
        elif self.source_algorithm == "ME":
            self.update_source_model_me(flooring_fn=flooring_fn)
        else:
            raise ValueError(
                "{}-algorithm-based source model updates are not supported.".format(
                    self.source_algorithm
                )
            )

    def update_source_model_mm(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF bases, activations, and latent variables by MM algorithm.

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

        if self.partitioning:
            self.update_latent_mm()

        self.update_basis_mm(flooring_fn=flooring_fn)
        self.update_activation_mm(flooring_fn=flooring_fn)

    def update_source_model_me(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF bases, activations, and latent variables by ME algorithm.

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

        if self.partitioning:
            self.update_latent_me()

        self.update_basis_me(flooring_fn=flooring_fn)
        self.update_activation_me(flooring_fn=flooring_fn)

    def update_latent_mm(self) -> None:
        r"""Update latent variables in NMF by MM algorithm.

        Update :math:`z_{nk}` as follows:

        .. math::
            z_{nk}
            &\leftarrow\left[\frac{\displaystyle\sum_{i,j}\frac{t_{ik}v_{kj}}
            {\tilde{r}_{ijn}\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}
            |y_{ijn}|^{2}}{\displaystyle\sum_{i,j}\dfrac{t_{ik}v_{kj}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            \right]^{\frac{p}{p+2}}z_{nk} \\
            z_{nk}
            &\leftarrow\frac{z_{nk}}{\sum_{n'}z_{n'k}}, \\
            \tilde{r}_{ijn}
            &= \frac{\nu}{\nu+2}\left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}}
            + \frac{2}{\nu+2}|y_{ijn}|^{2}.
        """
        p = self.domain
        nu = self.dof

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Y2 = np.abs(Y) ** 2
        p_p2 = p / (p + 2)
        nu_nu2 = nu / (nu + 2)

        Z = self.latent
        T, V = self.basis, self.activation

        TV = T[:, :, np.newaxis] * V[np.newaxis, :, :]
        ZTV = self.reconstruct_nmf(T, V, latent=Z)

        ZTV2p = ZTV ** (2 / p)
        R_tilde = nu_nu2 * ZTV2p + (1 - nu_nu2) * Y2
        RZTV = R_tilde * ZTV
        TV_RZTV = TV[np.newaxis, :, :, :] / RZTV[:, :, np.newaxis, :]
        num = np.sum(TV_RZTV * Y2[:, :, np.newaxis, :], axis=(1, 3))

        TV_ZTV = TV[np.newaxis, :, :, :] / ZTV[:, :, np.newaxis, :]
        denom = np.sum(TV_ZTV, axis=(1, 3))

        Z = ((num / denom) ** p_p2) * Z
        Z = Z / Z.sum(axis=0)

        self.latent = Z

    def update_basis_mm(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF bases by MM algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Update :math:`t_{ikn}` as follows:

        .. math::
            t_{ik}
            &\leftarrow\left[
            \frac{\displaystyle\sum_{j,n}\frac{z_{nk}v_{kj}}
            {\tilde{r}_{ijn}\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}
            |y_{ijn}|^{2}}{\displaystyle\sum_{j,n}
            \dfrac{z_{nk}v_{kj}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            \right]^{\frac{p}{p+2}}t_{ik}, \\
            \tilde{r}_{ijn}
            &= \frac{\nu}{\nu+2}\left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}}
            + \frac{2}{\nu+2}|y_{ijn}|^{2},

        if ``partitioning=True``. Otherwise

        .. math::
            t_{ikn}
            &\leftarrow \left[\frac{\displaystyle\sum_{j}
            \dfrac{v_{kjn}}{\tilde{r}_{ijn}\sum_{k'}t_{ik'n}v_{k'jn}}|y_{ijn}|^{2}}
            {\displaystyle\sum_{j}\frac{v_{kjn}}{\sum_{k'}t_{ik'n}v_{k'jn}}}\right]
            ^{\frac{p}{p+2}}t_{ikn}, \\
            \tilde{r}_{ijn}
            &= \frac{\nu}{\nu+2}\left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}
            + \frac{2}{\nu+2}|y_{ijn}|^{2}.
        """
        p = self.domain
        nu = self.dof
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Y2 = np.abs(Y) ** 2
        p_p2 = p / (p + 2)
        nu_nu2 = nu / (nu + 2)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZV = Z[:, :, np.newaxis] * V[np.newaxis, :, :]
            ZTV = self.reconstruct_nmf(T, V, latent=Z)

            ZTV2p = ZTV ** (2 / p)
            R_tilde = nu_nu2 * ZTV2p + (1 - nu_nu2) * Y2
            RZTV = R_tilde * ZTV
            ZV_RZTV = ZV[:, np.newaxis, :, :] / RZTV[:, :, np.newaxis, :]
            num = np.sum(ZV_RZTV * Y2[:, :, np.newaxis, :], axis=(0, 3))

            ZV_ZTV = ZV[:, np.newaxis, :, :] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(ZV_ZTV, axis=(0, 3))
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)

            TV2p = TV ** (2 / p)
            R_tilde = nu_nu2 * TV2p + (1 - nu_nu2) * Y2
            RTV = R_tilde * TV
            V_RTV = V[:, np.newaxis, :, :] / RTV[:, :, np.newaxis, :]
            num = np.sum(V_RTV * Y2[:, :, np.newaxis, :], axis=3)

            V_TV = V[:, np.newaxis, :, :] / TV[:, :, np.newaxis, :]
            denom = np.sum(V_TV, axis=3)

        T = ((num / denom) ** p_p2) * T
        T = flooring_fn(T)

        self.basis = T

    def update_activation_mm(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF activations by MM algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Update :math:`v_{kjn}` as follows:

        .. math::
            v_{kj}
            &\leftarrow\left[\frac{\displaystyle\sum_{i,n}\frac{z_{nk}t_{ik}}
            {\tilde{r}_{ijn}\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}
            |y_{ijn}|^{2}}{\displaystyle\sum_{i,n}\dfrac{z_{nk}t_{ik}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            \right]^{\frac{p}{p+2}}v_{kj}, \\
            \tilde{r}_{ijn}
            &= \frac{\nu}{\nu+2}\left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}}
            + \frac{2}{\nu+2}|y_{ijn}|^{2},

        if ``partitioning=True``. Otherwise

        .. math::
            v_{kjn}
            &\leftarrow \left[\frac{\displaystyle\sum_{i}
            \dfrac{t_{ikn}}{\tilde{r}_{ijn}\sum_{k'}t_{ik'n}v_{k'jn}}|y_{ijn}|^{2}}
            {\displaystyle\sum_{i}\frac{t_{ikn}}{\sum_{k'}t_{ik'n}v_{k'jn}}}
            \right]^{\frac{p}{p+2}}v_{kjn}, \\
            \tilde{r}_{ijn}
            &= \frac{\nu}{\nu+2}\left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}
            + \frac{2}{\nu+2}|y_{ijn}|^{2}.
        """
        p = self.domain
        nu = self.dof
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Y2 = np.abs(Y) ** 2
        p_p2 = p / (p + 2)
        nu_nu2 = nu / (nu + 2)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZT = Z[:, np.newaxis, :] * T[np.newaxis, :, :]
            ZTV = self.reconstruct_nmf(T, V, latent=Z)

            ZTV2p = ZTV ** (2 / p)
            R_tilde = nu_nu2 * ZTV2p + (1 - nu_nu2) * Y2
            RZTV = R_tilde * ZTV
            ZT_RZTV = ZT[:, :, :, np.newaxis] / RZTV[:, :, np.newaxis, :]
            num = np.sum(ZT_RZTV * Y2[:, :, np.newaxis, :], axis=(0, 1))

            ZT_ZTV = ZT[:, :, :, np.newaxis] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(ZT_ZTV, axis=(0, 1))
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)

            TV2p = TV ** (2 / p)
            R_tilde = nu_nu2 * TV2p + (1 - nu_nu2) * Y2
            RTV = R_tilde * TV
            T_RTV = T[:, :, :, np.newaxis] / RTV[:, :, np.newaxis, :]
            num = np.sum(T_RTV * Y2[:, :, np.newaxis, :], axis=1)

            T_TV = T[:, :, :, np.newaxis] / TV[:, :, np.newaxis, :]
            denom = np.sum(T_TV, axis=1)

        V = ((num / denom) ** p_p2) * V
        V = flooring_fn(V)

        self.activation = V

    def update_latent_me(self) -> None:
        r"""Update latent variables in NMF by ME algorithm.

        Update :math:`z_{nk}` as follows:

        .. math::
            z_{nk}
            &\leftarrow\frac{\displaystyle\sum_{i,j}\frac{t_{ik}v_{kj}}
            {\tilde{r}_{ijn}\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}
            |y_{ijn}|^{2}}{\displaystyle\sum_{i,j}\dfrac{t_{ik}v_{kj}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            z_{nk} \\
            z_{nk}
            &\leftarrow\frac{z_{nk}}{\sum_{n'}z_{n'k}}, \\
            \tilde{r}_{ijn}
            &= \frac{\nu}{\nu+2}\sum_{k}z_{nk}t_{ik}v_{kj}+\frac{2}{\nu+2}|y_{ijn}|^{2}.
        """
        nu = self.dof

        if self.domain != 2:
            raise ValueError("Domain parameter is expected 2, but given {}.".format(self.domain))

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Y2 = np.abs(Y) ** 2
        nu_nu2 = nu / (nu + 2)

        Z = self.latent
        T, V = self.basis, self.activation

        TV = T[:, :, np.newaxis] * V[np.newaxis, :, :]
        ZTV = self.reconstruct_nmf(T, V, latent=Z)

        R_tilde = nu_nu2 * ZTV + (1 - nu_nu2) * Y2
        RZTV = R_tilde * ZTV
        TV_RZTV = TV[np.newaxis, :, :, :] / RZTV[:, :, np.newaxis, :]
        num = np.sum(TV_RZTV * Y2[:, :, np.newaxis, :], axis=(1, 3))

        TV_ZTV = TV[np.newaxis, :, :, :] / ZTV[:, :, np.newaxis, :]
        denom = np.sum(TV_ZTV, axis=(1, 3))

        Z = (num / denom) * Z
        Z = Z / Z.sum(axis=0)

        self.latent = Z

    def update_basis_me(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF bases by ME algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Update :math:`t_{ikn}` as follows:

        .. math::
            t_{ik}
            &\leftarrow
            \frac{\displaystyle\sum_{j,n}\frac{z_{nk}v_{kj}}
            {\tilde{r}_{ijn}\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}
            |y_{ijn}|^{2}}{\displaystyle\sum_{j,n}
            \dfrac{z_{nk}v_{kj}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            t_{ik}, \\
            \tilde{r}_{ijn}
            &= \frac{\nu}{\nu+2}\sum_{k}z_{nk}t_{ik}v_{kj}+\frac{2}{\nu+2}|y_{ijn}|^{2},

        if ``partitioning=True``. Otherwise

        .. math::
            t_{ikn}
            &\leftarrow\frac{\displaystyle\sum_{j}
            \dfrac{v_{kjn}}{\tilde{r}_{ijn}\sum_{k'}t_{ik'n}v_{k'jn}}|y_{ijn}|^{2}}
            {\displaystyle\sum_{j}\frac{v_{kjn}}{\sum_{k'}t_{ik'n}v_{k'jn}}}
            t_{ikn}, \\
            \tilde{r}_{ijn}
            &= \frac{\nu}{\nu+2}\sum_{k}t_{ikn}v_{kjn}+\frac{2}{\nu+2}|y_{ijn}|^{2}.
        """
        nu = self.dof
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.domain != 2:
            raise ValueError("Domain parameter is expected 2, but given {}.".format(self.domain))

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Y2 = np.abs(Y) ** 2
        nu_nu2 = nu / (nu + 2)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZV = Z[:, :, np.newaxis] * V[np.newaxis, :, :]
            ZTV = self.reconstruct_nmf(T, V, latent=Z)

            R_tilde = nu_nu2 * ZTV + (1 - nu_nu2) * Y2
            RZTV = R_tilde * ZTV
            ZV_RZTV = ZV[:, np.newaxis, :, :] / RZTV[:, :, np.newaxis, :]
            num = np.sum(ZV_RZTV * Y2[:, :, np.newaxis, :], axis=(0, 3))

            ZV_ZTV = ZV[:, np.newaxis, :, :] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(ZV_ZTV, axis=(0, 3))
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)

            R_tilde = nu_nu2 * TV + (1 - nu_nu2) * Y2
            RTV = R_tilde * TV
            V_RTV = V[:, np.newaxis, :, :] / RTV[:, :, np.newaxis, :]
            num = np.sum(V_RTV * Y2[:, :, np.newaxis, :], axis=3)

            V_TV = V[:, np.newaxis, :, :] / TV[:, :, np.newaxis, :]
            denom = np.sum(V_TV, axis=3)

        T = (num / denom) * T
        T = flooring_fn(T)

        self.basis = T

    def update_activation_me(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF activations by ME algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Update :math:`v_{kjn}` as follows:

        .. math::
            v_{kj}
            &\leftarrow\frac{\displaystyle\sum_{i,n}\frac{z_{nk}t_{ik}}
            {\tilde{r}_{ijn}\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}
            |y_{ijn}|^{2}}{\displaystyle\sum_{i,n}\dfrac{z_{nk}t_{ik}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            v_{kj}, \\
            \tilde{r}_{ijn}
            &= \frac{\nu}{\nu+2}\sum_{k}z_{nk}t_{ik}v_{kj}+\frac{2}{\nu+2}|y_{ijn}|^{2},

        if ``partitioning=True``. Otherwise

        .. math::
            v_{kjn}
            &\leftarrow\frac{\displaystyle\sum_{i}
            \dfrac{t_{ikn}}{\tilde{r}_{ijn}\sum_{k'}t_{ik'n}v_{k'jn}}|y_{ijn}|^{2}}
            {\displaystyle\sum_{i}\frac{t_{ikn}}{\sum_{k'}t_{ik'n}v_{k'jn}}}
            v_{kjn}, \\
            \tilde{r}_{ijn}
            &= \frac{\nu}{\nu+2}\sum_{k}t_{ikn}v_{kjn}+\frac{2}{\nu+2}|y_{ijn}|^{2}.
        """
        nu = self.dof
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.domain != 2:
            raise ValueError("Domain parameter is expected 2, but given {}.".format(self.domain))

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Y2 = np.abs(Y) ** 2
        nu_nu2 = nu / (nu + 2)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZT = Z[:, np.newaxis, :] * T[np.newaxis, :, :]
            ZTV = self.reconstruct_nmf(T, V, latent=Z)

            R_tilde = nu_nu2 * ZTV + (1 - nu_nu2) * Y2
            RZTV = R_tilde * ZTV
            ZT_RZTV = ZT[:, :, :, np.newaxis] / RZTV[:, :, np.newaxis, :]
            num = np.sum(ZT_RZTV * Y2[:, :, np.newaxis, :], axis=(0, 1))

            ZT_ZTV = ZT[:, :, :, np.newaxis] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(ZT_ZTV, axis=(0, 1))
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)

            R_tilde = nu_nu2 * TV + (1 - nu_nu2) * Y2
            RTV = R_tilde * TV
            T_RTV = T[:, :, :, np.newaxis] / RTV[:, :, np.newaxis, :]
            num = np.sum(T_RTV * Y2[:, :, np.newaxis, :], axis=1)

            T_TV = T[:, :, :, np.newaxis] / TV[:, :, np.newaxis, :]
            denom = np.sum(T_TV, axis=1)

        V = (num / denom) * V
        V = flooring_fn(V)

        self.activation = V

    def update_spatial_model(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update demixing filters once.

        - If ``spatial_algorithm`` is ``IP`` or ``IP1``, ``update_spatial_model_ip1`` is called.
        - If ``spatial_algorithm`` is ``ISS`` or ``ISS1``, ``update_spatial_model_iss1`` is called.
        - If ``spatial_algorithm`` is ``IP2``, ``update_spatial_model_ip2`` is called.
        - If ``spatial_algorithm`` is ``ISS2``, ``update_spatial_model_iss2`` is called.

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
            self.update_spatial_model_ip1(flooring_fn=flooring_fn)
        elif self.spatial_algorithm in ["IP2"]:
            self.update_spatial_model_ip2(flooring_fn=flooring_fn)
        elif self.spatial_algorithm in ["ISS", "ISS1"]:
            self.update_spatial_model_iss1(flooring_fn=flooring_fn)
        elif self.spatial_algorithm in ["ISS2"]:
            self.update_spatial_model_iss2(flooring_fn=flooring_fn)
        else:
            raise NotImplementedError("Not support {}.".format(self.spatial_algorithm))

    def update_spatial_model_ip1(
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

        Demixing filters are updated sequentially for :math:`n=1,\ldots,N` as follows:

        .. math::
            \boldsymbol{w}_{in}
            &\leftarrow\left(\boldsymbol{W}_{in}^{\mathsf{H}}\boldsymbol{U}_{in}\right)^{-1} \
            \boldsymbol{e}_{n}, \\
            \boldsymbol{w}_{in}
            &\leftarrow\frac{\boldsymbol{w}_{in}}
            {\sqrt{\boldsymbol{w}_{in}^{\mathsf{H}}\boldsymbol{U}_{in}\boldsymbol{w}_{in}}},

        where

        .. math::
            \boldsymbol{U}_{in}
            = \frac{1}{J}\sum_{j}
            \frac{1}{\tilde{r}_{ijn}}\boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}.

        :math:`\tilde{r}_{ijn}` is defined as

        .. math::
            \tilde{r}_{ijn}
            = \frac{\nu}{\nu+2}\left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}}
            + \frac{2}{\nu+2}|y_{ijn}|^{2},

        if ``partitioning=True``. Otherwise

        .. math::
            \tilde{r}_{ijn}
            = \frac{\nu}{\nu+2}\left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}
            + \frac{2}{\nu+2}|y_{ijn}|^{2}.
        """
        p = self.domain
        nu = self.dof

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        Y2 = np.abs(Y) ** 2
        nu_nu2 = nu / (nu + 2)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            ZTV2p = ZTV ** (2 / p)
            R_tilde = nu_nu2 * ZTV2p + (1 - nu_nu2) * Y2
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)
            TV2p = TV ** (2 / p)
            R_tilde = nu_nu2 * TV2p + (1 - nu_nu2) * Y2

        varphi = 1 / R_tilde

        XX_Hermite = X[:, np.newaxis, :, :] * X[np.newaxis, :, :, :].conj()
        XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)

        varphi = varphi.transpose(1, 0, 2)
        varphi_XX = varphi[:, :, np.newaxis, np.newaxis, :] * XX_Hermite[:, np.newaxis, :, :, :]
        U = np.mean(varphi_XX, axis=-1)

        self.demix_filter = update_by_ip1(W, U, flooring_fn=flooring_fn)

    def update_spatial_model_ip2(
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
        compute weighted covariance matrix as follows:

        .. math::
            \boldsymbol{U}_{in}
            = \frac{1}{J}\sum_{j}
            \frac{1}{\tilde{r}_{ijn}}\boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}},

        :math:`\tilde{r}_{ijn}` is computed by

        .. math::
            \tilde{r}_{ijn}
            = \frac{\nu}{\nu+2}\left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}}
            + \frac{2}{\nu+2}|y_{ijn}|^{2},

        if ``partitioning=True``. \
        Otherwise,

        .. math::
            \tilde{r}_{ijn}
            = \frac{\nu}{\nu+2}\left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}
            + \frac{2}{\nu+2}|y_{ijn}|^{2}.

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
            &\leftarrow \boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}\boldsymbol{h}_{in_{2}}

        At each iteration, we update pairs of :math:`n_{1}` and :math:`n_{1}`
        for :math:`n_{1}\neq n_{2}`.
        """
        nu = self.dof
        p = self.domain
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        X, W = self.input, self.demix_filter

        nu_nu2 = nu / (nu + 2)
        Y = self.separate(X, demix_filter=W)
        Y2 = np.abs(Y) ** 2

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            ZTV2p = ZTV ** (2 / p)
            R_tilde = nu_nu2 * ZTV2p + (1 - nu_nu2) * Y2
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)
            TV2p = TV ** (2 / p)
            R_tilde = nu_nu2 * TV2p + (1 - nu_nu2) * Y2

        varphi = 1 / R_tilde

        XX_Hermite = X[:, np.newaxis, :, :] * X[np.newaxis, :, :, :].conj()
        XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)

        varphi = varphi.transpose(1, 0, 2)
        varphi_XX = varphi[:, :, np.newaxis, np.newaxis, :] * XX_Hermite[:, np.newaxis, :, :, :]
        U = np.mean(varphi_XX, axis=-1)

        self.demix_filter = update_by_ip2(
            W, U, flooring_fn=flooring_fn, pair_selector=self.pair_selector
        )

    def update_spatial_model_iss1(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update estimated spectrograms once using iterative source steering.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Update :math:`y_{ijn}` as follows:

        .. math::
            \boldsymbol{y}_{ij}
            & \leftarrow\boldsymbol{y}_{ij} - \boldsymbol{d}_{in}y_{ijn} \\
            d_{inn'}
            &= \begin{cases}
                \dfrac{\displaystyle\sum_{j}\dfrac{1}{\tilde{r}_{ijn}}
                y_{ijn'}y_{ijn}^{*}}{\displaystyle\sum_{j}\dfrac{1}
                {\tilde{r}_{ijn}}|y_{ijn}|^{2}}
                & (n'\neq n) \\
                1 - \dfrac{1}{\sqrt{\displaystyle\dfrac{1}{J}\sum_{j}\dfrac{1}
                {\tilde{r}_{ijn}}|y_{ijn}|^{2}}}
                & (n'=n)
            \end{cases}.

        :math:`\tilde{r}_{ijn}` is defined as

        .. math::
            \tilde{r}_{ijn}
            = \frac{\nu}{\nu+2}\left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}}
            + \frac{2}{\nu+2}|y_{ijn}|^{2},

        if ``partitioning=True``. Otherwise

        .. math::
            \tilde{r}_{ijn}
            = \frac{\nu}{\nu+2}\left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}
            + \frac{2}{\nu+2}|y_{ijn}|^{2}.
        """
        p = self.domain
        nu = self.dof
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        Y = self.output
        Y2 = np.abs(Y) ** 2
        nu_nu2 = nu / (nu + 2)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            ZTV2p = ZTV ** (2 / p)
            R_tilde = nu_nu2 * ZTV2p + (1 - nu_nu2) * Y2
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)
            TV2p = TV ** (2 / p)
            R_tilde = nu_nu2 * TV2p + (1 - nu_nu2) * Y2

        varphi = 1 / R_tilde

        self.output = update_by_iss1(Y, varphi, flooring_fn=flooring_fn)

    def update_spatial_model_iss2(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update estimated spectrograms once using pairwise iterative source steering.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Compute :math:`\boldsymbol{G}_{in}^{(n_{1},n_{2})}`
        and :math:`\boldsymbol{f}_{in}^{(n_{1},n_{2})}` for :math:`n_{1}\neq n_{2}`:

        .. math::
            \begin{array}{rclc}
                \boldsymbol{G}_{in}^{(n_{1},n_{2})}
                &=& {\displaystyle\frac{1}{J}\sum_{j}}\dfrac{1}{\tilde{r}_{ijn}}
                \boldsymbol{y}_{ij}^{(n_{1},n_{2})}{\boldsymbol{y}_{ij}^{(n_{1},n_{2})}}^{\mathsf{H}}
                &(n=1,\ldots,N), \\
                \boldsymbol{f}_{in}^{(n_{1},n_{2})}
                &=& {\displaystyle\frac{1}{J}\sum_{j}}
                \dfrac{1}{\tilde{r}_{ijn}}y_{ijn}^{*}\boldsymbol{y}_{ij}^{(n_{1},n_{2})}
                &(n\neq n_{1},n_{2}),
            \end{array}

        where

        .. math::
            \tilde{r}_{ijn}
            = \frac{\nu}{\nu+2}\left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}}
            + \frac{2}{\nu+2}|y_{ijn}|^{2}

        if ``partitioning=True``.
        Otherwise,

        .. math::
            \tilde{r}_{ijn}
            = \frac{\nu}{\nu+2}\left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}
            + \frac{2}{\nu+2}|y_{ijn}|^{2}.

        Using :math:`\boldsymbol{G}_{in}^{(n_{1},n_{2})}` and
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

        where :math:`\boldsymbol{h}_{in}` (:math:`n=n_{1},n_{2}`) is
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
        """
        p = self.domain
        nu = self.dof
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        Y = self.output
        Y2 = np.abs(Y) ** 2
        nu_nu2 = nu / (nu + 2)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            ZTV2p = ZTV ** (2 / p)
            R_tilde = nu_nu2 * ZTV2p + (1 - nu_nu2) * Y2
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)
            TV2p = TV ** (2 / p)
            R_tilde = nu_nu2 * TV2p + (1 - nu_nu2) * Y2

        varphi = 1 / R_tilde

        self.output = update_by_iss2(
            Y, varphi, flooring_fn=flooring_fn, pair_selector=self.pair_selector
        )

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L}
            = \frac{1}{J}\sum_{i,j}
            \left\{1+\frac{\nu}{2}\log\left(1+\frac{2}{\nu}
            \frac{|y_{ijn}|^{2}}{r_{ijn}}\right)
            + \log r_{ijn}\right\}
            -2\sum_{i}\log\left|\det\boldsymbol{W}_{i}\right|,

        where

        .. math::
            r_{ijn}
            = \left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}},

        if ``partitioning=True``, otherwise

        .. math::
            r_{ijn}
            = \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}.

        Returns:
            Computed loss.
        """
        nu = self.dof
        p = self.domain

        if self.demix_filter is None:
            X, Y = self.input, self.output
            Y2 = np.abs(Y) ** 2
            X, Y = X.transpose(1, 0, 2), Y.transpose(1, 0, 2)
            X_Hermite = X.transpose(0, 2, 1).conj()
            XX_Hermite = X @ X_Hermite
            W = Y @ X_Hermite @ np.linalg.inv(XX_Hermite)
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
            Y2 = np.abs(Y) ** 2

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation
            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            Y2ZTV2p = Y2 / (ZTV ** (2 / p))
            loss = (1 + nu / 2) * np.log(1 + (2 / nu) * Y2ZTV2p) + (2 / p) * np.log(ZTV)
        else:
            T, V = self.basis, self.activation
            TV = self.reconstruct_nmf(T, V)
            Y2TV2p = Y2 / (TV ** (2 / p))
            loss = (1 + nu / 2) * np.log(1 + (2 / nu) * Y2TV2p) + (2 / p) * np.log(TV)

        logdet = self.compute_logdet(W)  # (n_bins,)

        loss = np.sum(loss.mean(axis=-1), axis=0) - 2 * logdet
        loss = loss.sum(axis=0).item()

        return loss

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


class GGDILRMA(ILRMABase):
    r"""Independent low-rank matrix analysis (ILRMA) on a generalized Gaussian distribution.

    We assume :math:`y_{ijn}` follows a generalized Gaussian distribution.

    .. math::
        p(y_{ijn})
        = \frac{\beta}{2\pi r_{ijn}\Gamma\left(\frac{2}{\beta}\right)}
        \exp\left\{-\left(\frac{|y_{ijn}|^{2}}{r_{ijn}}\right)^{\frac{\beta}{2}}\right\},

    where

    .. math::
        r_{ijn}
        = \left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}},

    if ``partitioning=True``. Otherwise,

    .. math::
        r_{ijn}
        = \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}.

    :math:`\beta` is a shape parameter of a generalized Gaussian distribution.

    Args:
        n_basis (int):
            Number of NMF bases.
        beta (float):
            Shape parameter in generalized Gaussian distribution.
        spatial_algorithm (str):
            Algorithm for demixing filter updates.
            Choose ``IP``, ``IP1``, ``IP2``, ``ISS``, ``ISS1``, or ``ISS2``.
            Default: ``IP``.
        source_algorithm (str):
            Algorithm for source model updates.
            Only ``MM`` is supported: Default: ``MM``.
        domain (float):
            Domain parameter. Default: ``2``.
        partitioning (bool):
            Whether to use partioning function. Default: ``False``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        pair_selector (callable, optional):
            Selector to choose updaing pair in ``IP2`` and ``ISS2``.
            If ``None`` is given, ``sequential_pair_selector`` is used.
            Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        normalization (bool or str, optional):
            Normalization of demixing filters and NMF parameters.
            Choose ``power`` or ``projection_back``.
            Default: ``power``.
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
        rng (numpy.random.Generator, optioinal):
            Random number generator. This is mainly used to randomly initialize NMF.
            If ``None`` is given, ``np.random.default_rng()`` is used.
            Default: ``None``.

    Examples:
        Update demixing filters by IP:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> ilrma = GGDILRMA(
            ...     n_basis=2,
            ...     beta=1.99,
            ...     spatial_algorithm="IP",
            ...     rng=np.random.default_rng(42),
            ... )
            >>> spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by IP2:

        .. code-block:: python

            >>> from ssspy.utils.select_pair import sequential_pair_selector

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> ilrma = GGDILRMA(
            ...     n_basis=2,
            ...     beta=1.99,
            ...     spatial_algorithm="IP2",
            ...     pair_selector=sequential_pair_selector,
            ...     rng=np.random.default_rng(42),
            ... )
            >>> spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by ISS:

        .. code-block:: python

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> ilrma = GGDILRMA(
            ...     n_basis=2,
            ...     beta=1.99,
            ...     spatial_algorithm="ISS",
            ...     rng=np.random.default_rng(42),
            ... )
            >>> spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)

        Update demixing filters by ISS2:

        .. code-block:: python

            >>> import functools
            >>> from ssspy.utils.select_pair import sequential_pair_selector

            >>> n_channels, n_bins, n_frames = 2, 2049, 128
            >>> spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
            ...     + 1j * np.random.randn(n_channels, n_bins, n_frames)

            >>> ilrma = GGDILRMA(
            ...     n_basis=2,
            ...     beta=1.99,
            ...     spatial_algorithm="ISS2",
            ...     pair_selector=functools.partial(sequential_pair_selector, step=2),
            ...     rng=np.random.default_rng(42),
            ... )
            >>> spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
            >>> print(spectrogram_mix.shape, spectrogram_est.shape)
            (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        n_basis: int,
        beta: float,
        spatial_algorithm: str = "IP",
        source_algorithm: str = "MM",
        domain: float = 2,
        partitioning: bool = False,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        pair_selector: Optional[Callable[[int], Iterable[Tuple[int, int]]]] = None,
        callbacks: Optional[
            Union[Callable[["GGDILRMA"], None], List[Callable[["GGDILRMA"], None]]]
        ] = None,
        normalization: Optional[Union[bool, str]] = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            n_basis=n_basis,
            partitioning=partitioning,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
            rng=rng,
        )

        assert 0 < beta < 2, "Shape parameter {} shoule be chosen from (0, 2).".format(beta)
        assert spatial_algorithm in spatial_algorithms, "Not support {}.".format(spatial_algorithms)
        assert source_algorithm == "MM", "Not support {}.".format(source_algorithm)
        assert 0 < domain <= 2, "domain parameter should be chosen from [0, 2]."

        if spatial_algorithm == "IPA":
            raise ValueError("IPA is not supported for GGD-ILRMA.")

        self.beta = beta
        self.spatial_algorithm = spatial_algorithm
        self.source_algorithm = source_algorithm
        self.domain = domain
        self.normalization = normalization

        if pair_selector is None:
            if spatial_algorithm in ["IP2", "ISS2"]:
                self.pair_selector = sequential_pair_selector
        else:
            self.pair_selector = pair_selector

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

        self._reset(flooring_fn=self.flooring_fn, **kwargs)

        # Call __call__ of ILRMABase's parent, i.e. __call__ of IterativeMethodBase
        super(ILRMABase, self).__call__(n_iter=n_iter, initial_call=initial_call)

        if self.scale_restoration:
            self.restore_scale()

        if self.demix_filter is None:
            pass
        else:
            self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "GGDILRMA("
        s += "n_basis={n_basis}"
        s += ", beta={beta}"
        s += ", spatial_algorithm={spatial_algorithm}"
        s += ", source_algorithm={source_algorithm}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalization={normalization}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
        **kwargs,
    ) -> None:
        r"""Reset attributes by given keyword arguments.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.
            kwargs:
                Keyword arguments to set as attributes of ILRMA.
        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        super()._reset(flooring_fn=flooring_fn, **kwargs)

        if self.spatial_algorithm in ["ISS", "ISS1", "ISS2"]:
            self.demix_filter = None

    def update_once(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF parameters and demixing filters once.

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

        self.update_source_model(flooring_fn=flooring_fn)
        self.update_spatial_model(flooring_fn=flooring_fn)

        if self.normalization:
            self.normalize(flooring_fn=flooring_fn)

    def update_source_model(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF bases, activations, and latent variables.

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

        if self.source_algorithm == "MM":
            self.update_source_model_mm(flooring_fn=flooring_fn)
        else:
            raise ValueError(
                "{}-algorithm-based source model updates are not supported.".format(
                    self.source_algorithm
                )
            )

    def update_source_model_mm(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF bases, activations, and latent variables by MM algorithm.

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

        if self.partitioning:
            self.update_latent_mm()

        self.update_basis_mm(flooring_fn=flooring_fn)
        self.update_activation_mm(flooring_fn=flooring_fn)

    def update_latent_mm(self) -> None:
        r"""Update latent variables in NMF by MM algorithm.

        Update :math:`z_{nk}` as follows:

        .. math::
            z_{nk}
            &\leftarrow\left[
            \frac{\beta}{2}
            \frac{\displaystyle\sum_{i,j}\frac{t_{ik}v_{kj}}
            {(\sum_{k'}z_{nk'}t_{ik'}v_{k'j})^{\frac{\beta+p}{2}}}|y_{ijn}|^{\beta}}
            {\displaystyle\sum_{i,j}\frac{t_{ik}v_{kj}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            \right]^{\frac{p}{\beta+p}}z_{nk}, \\
            z_{nk}
            &\leftarrow\frac{z_{nk}}{\displaystyle\sum_{n'}z_{n'k}}.
        """
        p = self.domain
        beta = self.beta

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Yb = np.abs(Y) ** beta
        p_bp = p / (beta + p)
        bp_p = (beta + p) / p

        Z = self.latent
        T, V = self.basis, self.activation

        TV = T[:, :, np.newaxis] * V[np.newaxis, :, :]
        ZTV = self.reconstruct_nmf(T, V, latent=Z)

        ZTVbpp = ZTV**bp_p
        TV_RZTV = TV[np.newaxis, :, :, :] / ZTVbpp[:, :, np.newaxis, :]
        num = (beta / 2) * np.sum(TV_RZTV * Yb[:, :, np.newaxis, :], axis=(1, 3))

        TV_ZTV = TV[np.newaxis, :, :, :] / ZTV[:, :, np.newaxis, :]
        denom = np.sum(TV_ZTV, axis=(1, 3))

        Z = ((num / denom) ** p_bp) * Z
        Z = Z / Z.sum(axis=0)

        self.latent = Z

    def update_basis_mm(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF bases by MM algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Update :math:`t_{ikn}` as follows:

        .. math::
            t_{ik}
            \leftarrow\left[
            \frac{\beta}{2}
            \frac{\displaystyle\sum_{j,n}\frac{z_{nk}v_{kj}}
            {(\sum_{k'}z_{nk'}t_{ik'}v_{k'j})^{\frac{\beta+p}{p}}}|y_{ijn}|^{\beta}}
            {\displaystyle\sum_{j,n}\frac{z_{nk}v_{kj}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            \right]^{\frac{p}{\beta+p}}t_{ik},

        if ``partitioning=True``. Otherwise

        .. math::
            t_{ikn}
            \leftarrow\left[
            \frac{\beta}{2}
            \frac{\displaystyle\sum_{j}\frac{v_{kjn}}
            {(\sum_{k'}t_{ik'n}v_{k'jn})^{\frac{\beta+p}{p}}}|y_{ijn}|^{\beta}}
            {\displaystyle\sum_{j}\frac{v_{kjn}}{\sum_{k'}t_{ik'n}v_{k'jn}}}
            \right]^{\frac{p}{\beta+p}}t_{ikn}.
        """
        p = self.domain
        beta = self.beta
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Yb = np.abs(Y) ** beta
        p_bp = p / (beta + p)
        bp_p = (beta + p) / p

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZV = Z[:, :, np.newaxis] * V[np.newaxis, :, :]
            ZTV = self.reconstruct_nmf(T, V, latent=Z)

            ZTVbpp = ZTV**bp_p
            ZV_ZTVbpp = ZV[:, np.newaxis, :, :] / ZTVbpp[:, :, np.newaxis, :]
            num = (beta / 2) * np.sum(ZV_ZTVbpp * Yb[:, :, np.newaxis, :], axis=(0, 3))

            ZV_ZTV = ZV[:, np.newaxis, :, :] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(ZV_ZTV, axis=(0, 3))
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)

            TVbpp = TV**bp_p
            V_TVbpp = V[:, np.newaxis, :, :] / TVbpp[:, :, np.newaxis, :]
            num = (beta / 2) * np.sum(V_TVbpp * Yb[:, :, np.newaxis, :], axis=3)

            V_TV = V[:, np.newaxis, :, :] / TV[:, :, np.newaxis, :]
            denom = np.sum(V_TV, axis=3)

        T = ((num / denom) ** p_bp) * T
        T = flooring_fn(T)

        self.basis = T

    def update_activation_mm(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update NMF activations by MM algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Update :math:`v_{kjn}` as follows:

        .. math::
            v_{kj}
            \leftarrow\left[
            \frac{\beta}{2}
            \frac{\displaystyle\sum_{i,n}\frac{z_{nk}t_{ik}}
            {(\sum_{k'}z_{nk'}t_{ik'}v_{k'j})^{\frac{\beta+p}{p}}}|y_{ijn}|^{\beta}}
            {\displaystyle\sum_{i,n}\frac{z_{nk}t_{ik}}{\sum_{k'}z_{nk'}t_{ik'}v_{k'j}}}
            \right]^{\frac{p}{\beta+p}}v_{kj},

        if ``partitioning=True``. Otherwise

        .. math::
            v_{kj}
            \leftarrow\left[
            \frac{\beta}{2}
            \frac{\displaystyle\sum_{i}\frac{t_{ikn}}
            {(\sum_{k'}t_{ik'n}v_{k'jn})^{\frac{\beta+p}{p}}}|y_{ijn}|^{\beta}}
            {\displaystyle\sum_{i}\frac{t_{ik}}{\sum_{k'}t_{ik'n}v_{k'jn}}}
            \right]^{\frac{p}{\beta+p}}v_{kjn}.
        """
        p = self.domain
        beta = self.beta
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.demix_filter is None:
            Y = self.output
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        Yb = np.abs(Y) ** beta
        p_bp = p / (beta + p)
        bp_p = (beta + p) / p

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZT = Z[:, np.newaxis, :] * T[np.newaxis, :, :]
            ZTV = self.reconstruct_nmf(T, V, latent=Z)

            ZTVbpp = ZTV**bp_p
            ZT_ZTVbpp = ZT[:, :, :, np.newaxis] / ZTVbpp[:, :, np.newaxis, :]
            num = (beta / 2) * np.sum(ZT_ZTVbpp * Yb[:, :, np.newaxis, :], axis=(0, 1))

            ZT_ZTV = ZT[:, :, :, np.newaxis] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(ZT_ZTV, axis=(0, 1))
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)

            TVbpp = TV**bp_p
            T_TVbpp = T[:, :, :, np.newaxis] / TVbpp[:, :, np.newaxis, :]
            num = (beta / 2) * np.sum(T_TVbpp * Yb[:, :, np.newaxis, :], axis=1)

            T_TV = T[:, :, :, np.newaxis] / TV[:, :, np.newaxis, :]
            denom = np.sum(T_TV, axis=1)

        V = ((num / denom) ** p_bp) * V
        V = flooring_fn(V)

        self.activation = V

    def update_spatial_model(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update demixing filters once.

        - If ``spatial_algorithm`` is ``IP`` or ``IP1``, ``update_spatial_model_ip1`` is called.
        - If ``spatial_algorithm`` is ``ISS`` or ``ISS1``, ``update_spatial_model_iss1`` is called.
        - If ``spatial_algorithm`` is ``IP2``, ``update_spatial_model_ip2`` is called.
        - If ``spatial_algorithm`` is ``ISS2``, ``update_spatial_model_iss2`` is called.

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
            self.update_spatial_model_ip1(flooring_fn=flooring_fn)
        elif self.spatial_algorithm in ["IP2"]:
            self.update_spatial_model_ip2(flooring_fn=flooring_fn)
        elif self.spatial_algorithm in ["ISS", "ISS1"]:
            self.update_spatial_model_iss1(flooring_fn=flooring_fn)
        elif self.spatial_algorithm in ["ISS2"]:
            self.update_spatial_model_iss2(flooring_fn=flooring_fn)
        else:
            raise NotImplementedError("Not support {}.".format(self.spatial_algorithm))

    def update_spatial_model_ip1(
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

        Demixing filters are updated sequentially for :math:`n=1,\ldots,N` as follows:

        .. math::
            \boldsymbol{w}_{in}
            &\leftarrow\left(\boldsymbol{W}_{in}^{\mathsf{H}}\boldsymbol{U}_{in}\right)^{-1} \
            \boldsymbol{e}_{n}, \\
            \boldsymbol{w}_{in}
            &\leftarrow\frac{\boldsymbol{w}_{in}}
            {\sqrt{\boldsymbol{w}_{in}^{\mathsf{H}}\boldsymbol{U}_{in}\boldsymbol{w}_{in}}},

        where

        .. math::
            \boldsymbol{U}_{in}
            \leftarrow\frac{1}{J}\sum_{i,j,n}
            \frac{\boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}}{\tilde{r}_{ijn}}.

        :math:`\tilde{r}_{ijn}` is computed as

        .. math::
            \tilde{r}_{ijn}
            = \frac{2|y_{ijn}|^{2-\beta}}{\beta}
            \left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{\beta}{p}},

        if ``partitioning=True``. Otherwise,

        .. math::
            \tilde{r}_{ijn}
            = \frac{2|y_{ijn}|^{2-\beta}}{\beta}
            \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{\beta}{p}}.
        """
        p = self.domain
        beta = self.beta
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        Y2b = np.abs(Y) ** (2 - beta)
        Y2b = flooring_fn(Y2b)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            ZTVbp = ZTV ** (beta / p)
            R_tilde = (2 / beta) * Y2b * ZTVbp
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)
            TVbp = TV ** (beta / p)
            R_tilde = (2 / beta) * Y2b * TVbp

        varphi = 1 / R_tilde

        XX_Hermite = X[:, np.newaxis, :, :] * X[np.newaxis, :, :, :].conj()
        XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)

        varphi = varphi.transpose(1, 0, 2)
        varphi_XX = varphi[:, :, np.newaxis, np.newaxis, :] * XX_Hermite[:, np.newaxis, :, :, :]
        U = np.mean(varphi_XX, axis=-1)

        self.demix_filter = update_by_ip1(W, U, flooring_fn=flooring_fn)

    def update_spatial_model_ip2(
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
        compute weighted covariance matrix as follows:

        .. math::
            \boldsymbol{U}_{in}
            = \frac{1}{J}\sum_{j}
            \frac{1}{\tilde{r}_{ijn}}\boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}},

        :math:`\tilde{r}_{ijn}` is computed by

        .. math::
            \tilde{r}_{ijn}
            = \frac{2|y_{ijn}|^{2-\beta}}{\beta}
            \left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{\beta}{p}},

        if ``partitioning=True``. \
        Otherwise,

        .. math::
            \tilde{r}_{ijn}
            = \frac{2|y_{ijn}|^{2-\beta}}{\beta}
            \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{\beta}{p}}.

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
            &\leftarrow \boldsymbol{P}_{in_{2}}^{(n_{1},n_{2})}\boldsymbol{h}_{in_{2}}

        At each iteration, we update pairs of :math:`n_{1}` and :math:`n_{1}`
        for :math:`n_{1}\neq n_{2}`.
        """
        p = self.domain
        beta = self.beta
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        Y2b = np.abs(Y) ** (2 - beta)
        Y2b = flooring_fn(Y2b)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            ZTVbp = ZTV ** (beta / p)
            R_tilde = (2 / beta) * Y2b * ZTVbp
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)
            TVbp = TV ** (beta / p)
            R_tilde = (2 / beta) * Y2b * TVbp

        varphi = 1 / R_tilde

        XX_Hermite = X[:, np.newaxis, :, :] * X[np.newaxis, :, :, :].conj()
        XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)

        varphi = varphi.transpose(1, 0, 2)
        varphi_XX = varphi[:, :, np.newaxis, np.newaxis, :] * XX_Hermite[:, np.newaxis, :, :, :]
        U = np.mean(varphi_XX, axis=-1)

        self.demix_filter = update_by_ip2(
            W, U, flooring_fn=flooring_fn, pair_selector=self.pair_selector
        )

    def update_spatial_model_iss1(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update estimated spectrograms once using iterative source steering.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Update :math:`y_{ijn}` as follows:

        .. math::
            \boldsymbol{y}_{ij}
            & \leftarrow\boldsymbol{y}_{ij} - \boldsymbol{d}_{in}y_{ijn} \\
            d_{inn'}
            &= \begin{cases}
                \dfrac{\displaystyle\sum_{j}\dfrac{1}{\tilde{r}_{ijn}}
                y_{ijn'}y_{ijn}^{*}}{\displaystyle\sum_{j}\dfrac{1}
                {\tilde{r}_{ijn}}|y_{ijn}|^{2}}
                & (n'\neq n) \\
                1 - \dfrac{1}{\sqrt{\displaystyle\dfrac{1}{J}\sum_{j}\dfrac{1}
                {\tilde{r}_{ijn}}|y_{ijn}|^{2}}} & (n'=n)
            \end{cases},

        where :math:`\tilde{r}_{ijn}` is computed as

        .. math::
            \tilde{r}_{ijn}
            = \frac{2|y_{ijn}|^{2-\beta}}{\beta}
            \left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{\beta}{p}},

        if ``partitioning=True``. Otherwise,

        .. math::
            \tilde{r}_{ijn}
            = \frac{2|y_{ijn}|^{2-\beta}}{\beta}
            \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{\beta}{p}}.
        """
        p = self.domain
        beta = self.beta
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        Y = self.output
        Y2b = np.abs(Y) ** (2 - beta)
        Y2b = flooring_fn(Y2b)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            ZTVbp = ZTV ** (beta / p)
            R_bar = Y2b * ZTVbp
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)
            TVbp = TV ** (beta / p)
            R_bar = Y2b * TVbp

        varphi = beta / (2 * R_bar)

        self.output = update_by_iss1(Y, varphi, flooring_fn=flooring_fn)

    def update_spatial_model_iss2(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update estimated spectrograms once using pairwise iterative source steering.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Compute :math:`\boldsymbol{G}_{in}^{(n_{1},n_{2})}`
        and :math:`\boldsymbol{f}_{in}^{(n_{1},n_{2})}` for :math:`n_{1}\neq n_{2}`:

        .. math::
            \begin{array}{rclc}
                \boldsymbol{G}_{in}^{(n_{1},n_{2})}
                &=& {\displaystyle\frac{1}{J}\sum_{j}}\dfrac{1}{\tilde{r}_{ijn}}
                \boldsymbol{y}_{ij}^{(n_{1},n_{2})}{\boldsymbol{y}_{ij}^{(n_{1},n_{2})}}^{\mathsf{H}}
                &(n=1,\ldots,N), \\
                \boldsymbol{f}_{in}^{(n_{1},n_{2})}
                &=& {\displaystyle\frac{1}{J}\sum_{j}}
                \dfrac{1}{\tilde{r}_{ijn}}y_{ijn}^{*}\boldsymbol{y}_{ij}^{(n_{1},n_{2})}
                &(n\neq n_{1},n_{2}),
            \end{array}

        where

        .. math::
            \tilde{r}_{ijn}
            = \frac{2}{\beta}|y_{ijn}|^{2-\beta}
            \left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{\beta}{p}},

        if ``partitioning=True``.
        Otherwise,

        .. math::
            \tilde{r}_{ijn}
            = \frac{2}{\beta}|y_{ijn}|^{2-\beta}
            \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{\beta}{p}}.

        Using :math:`\boldsymbol{G}_{in}^{(n_{1},n_{2})}` and
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

        where :math:`\boldsymbol{h}_{in}` (:math:`n=n_{1},n_{2}`) is
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
        """
        p = self.domain
        beta = self.beta
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        Y = self.output
        Y2b = np.abs(Y) ** (2 - beta)
        Y2b = flooring_fn(Y2b)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            ZTVbp = ZTV ** (beta / p)
            R_tilde = (2 / beta) * Y2b * ZTVbp
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)
            TVbp = TV ** (beta / p)
            R_tilde = (2 / beta) * Y2b * TVbp

        varphi = 1 / R_tilde

        self.output = update_by_iss2(
            Y, varphi, flooring_fn=flooring_fn, pair_selector=self.pair_selector
        )

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L}
            = \frac{1}{J}\sum_{i,j,n}
            \left\{\left(\frac{|y_{ijn}|^{2}}{r_{ijn}}\right)^{\frac{\beta}{2}}
            + \log r_{ijn}\right\}
            - 2\sum_{i}\log|\det\boldsymbol{W}_{i}|,

        where

        .. math::
            r_{ijn}
            = \left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)^{\frac{2}{p}},

        if ``partitioning=True``. Otherwise

        .. math::
            r_{ijn}
            = \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}.

        Returns:
            Computed loss.
        """
        beta = self.beta
        p = self.domain

        if self.demix_filter is None:
            X, Y = self.input, self.output
            Yb = np.abs(Y) ** beta
            X, Y = X.transpose(1, 0, 2), Y.transpose(1, 0, 2)
            X_Hermite = X.transpose(0, 2, 1).conj()
            XX_Hermite = X @ X_Hermite
            W = Y @ X_Hermite @ np.linalg.inv(XX_Hermite)
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
            Yb = np.abs(Y) ** beta

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation
            ZTV = self.reconstruct_nmf(T, V, latent=Z)
            R = ZTV ** (beta / p)
            loss = Yb / R + (2 / p) * np.log(ZTV)
        else:
            T, V = self.basis, self.activation
            TV = self.reconstruct_nmf(T, V)
            R = TV ** (beta / p)
            loss = Yb / R + (2 / p) * np.log(TV)

        logdet = self.compute_logdet(W)  # (n_bins,)

        loss = np.sum(loss.mean(axis=-1), axis=0) - 2 * logdet
        loss = loss.sum(axis=0).item()

        return loss

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
