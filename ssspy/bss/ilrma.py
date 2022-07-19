from typing import Optional, Union, List, Tuple, Callable, Iterable
import functools
import warnings

import numpy as np

from ._flooring import max_flooring
from ._select_pair import sequential_pair_selector
from ..linalg import eigh
from ..algorithm import projection_back

__all__ = ["GaussILRMA", "TILRMA"]

algorithms_spatial = ["IP", "IP1", "IP2", "ISS", "ISS1", "ISS2"]
EPS = 1e-10


class ILRMAbase:
    r"""Base class of independent low-rank matrix analysis (ILRMA) [#kitamura2016determined]_.

    Args:
        n_basis (int):
            Number of NMF bases.
        partitioning (bool):
            Whether to use partioning function. Default: ``False``.
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
        rng (numpy.random.Generator):
            Random number generator. This is mainly used to randomly initialize NMF.
            Default: ``numpy.random.default_rng()``.

    .. [#kitamura2016determined]
        D. Kitamura et al.,
        "Determined blind source separation unifying independent vector analysis \
        and nonnegative matrix factorization,"
        *IEEE/ACM Trans. ASLP.*, vol. 24, no. 9, pp. 1626-1641, 2016.
    """

    def __init__(
        self,
        n_basis: int,
        partitioning: bool = False,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["ILRMAbase"], None], List[Callable[["ILRMAbase"], None]]]
        ] = None,
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        self.n_basis = n_basis
        self.partitioning = partitioning

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

        self.rng = rng

        self.should_record_loss = should_record_loss

        if self.should_record_loss:
            self.loss = []
        else:
            self.loss = None

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

        if self.should_apply_projection_back:
            self.apply_projection_back()

        if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
            self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "ILRMA("
        s += "n_basis={n_basis}"
        s += ", partitioning={partitioning}"
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
                Set arguments as attributes of ILRMA.
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

        self._init_nmf(rng=self.rng)

    def _init_nmf(self, rng: np.random.Generator = np.random.default_rng()) -> None:
        r"""Initialize NMF.

        Args:
            rng (numpy.random.Generator):
                Random number generator. Default: ``numpy.random.default_rng()``.
        """
        n_basis = self.n_basis
        n_sources = self.n_sources
        n_bins, n_frames = self.n_bins, self.n_frames

        if self.partitioning:
            Z = rng.random((n_sources, n_basis))
            Z = Z / Z.sum(axis=0)
            T = rng.random((n_bins, n_basis))
            V = rng.random((n_basis, n_frames))

            Z = self.flooring_fn(Z)
            T = self.flooring_fn(T)
            V = self.flooring_fn(V)

            self.latent = Z
            self.basis, self.activation = T, V
        else:
            T = rng.random((n_sources, n_bins, n_basis))
            V = rng.random((n_sources, n_basis, n_frames))

            T = self.flooring_fn(T)
            V = self.flooring_fn(V)

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
            numpy.ndarray:
                The separated signal in frequency-domain.
                The shape is (n_sources, n_bins, n_frames).
        """
        X, W = input, demix_filter
        Y = W @ X.transpose(1, 0, 2)
        output = Y.transpose(1, 0, 2)

        return output

    def reconstruct_nmf(
        self, basis: np.ndarray, activation: np.ndarray, latent: Optional[np.ndarray] = None
    ):
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
            numpy.ndarray:
                Reconstructed NMF.
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
        r"""Update demixing filters once.
        """
        raise NotImplementedError("Implement 'update_once' method.")

    def normalize(self) -> None:
        r"""Normalize demixing filters and NMF parameters.
        """
        normalization = self.normalization

        assert normalization, "Set normalization."

        if type(normalization) is bool:
            # when normalization is True
            normalization = "power"

        if normalization == "power":
            self.normalize_by_power()
        elif normalization == "projection_back":
            self.normalize_by_projection_back()
        else:
            raise NotImplementedError("Normalization {} is not implemented.".format(normalization))

    def normalize_by_power(self):
        r"""Normalize demixing filters and NMF parameters by power.

        Demixing filters are normalized by

        .. math::
            \boldsymbol{w}_{in}
            &\leftarrow\frac{\boldsymbol{w}_{in}}{\psi_{in}},

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

        if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
        else:
            Y = self.output

        Y2 = np.mean(np.abs(Y) ** 2, axis=(-2, -1))
        psi = np.sqrt(Y2)
        psi = self.flooring_fn(psi)

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

        if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
            W = self.demix_filter
            W = W / psi[np.newaxis, :, np.newaxis]
            self.demix_filter = W
        else:
            Y = Y / psi[:, np.newaxis, np.newaxis]
            self.output = Y

    def normalize_by_projection_back(self):
        r"""Normalize demixing filters and NMF parameters by projection back.

        Demixing filters are normalized by

        .. math::
            \boldsymbol{w}_{in}
            &\leftarrow\frac{\boldsymbol{w}_{in}}{\psi_{in}},

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

            if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
                W = self.demix_filter

                scale = np.linalg.inv(W)
                scale = scale[:, reference_id, :]
                W = W * scale[:, :, np.newaxis]

                self.demix_filter = W
            else:
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

            scale = scale.transpose(1, 0)
            scale = np.abs(scale) ** p
            T = T * scale[:, :, np.newaxis]

            self.basis = T

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        Returns:
            float:
                Computed loss.
        """
        raise NotImplementedError("Implement 'compute_loss' method.")

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


class GaussILRMA(ILRMAbase):
    r"""Independent low-rank matrix analysis (ILRMA) on Gaussian distribution.

    Args:
        n_basis (int):
            Number of NMF bases.
        algorithm_spatial (str):
            Algorithm for demixing filter updates.
            Choose "IP", "IP1", "IP2", "ISS", "ISS1", or "ISS2".
            Default: "IP".
        domain (float):
            Domain parameter. Default: ``2``.
        partitioning (bool):
            Whether to use partioning function. Default: ``False``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        pair_selector (callable, optional):
            Selector to choose updaing pair in ``IP2`` and ``ISS2``.
            If ``None`` is given, ``partial(sequential_pair_selector, sort=True)`` is used.
            Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        normalization (bool or str, optional):
            Normalization of demixing filters and NMF parameters.
            Choose "power" or "projection_back".
            Default: ``"power"``.
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
        rng (numpy.random.Generator):
            Random number generator. This is mainly used to randomly initialize NMF.
            Default: ``numpy.random.default_rng()``.
    """

    def __init__(
        self,
        n_basis: int,
        algorithm_spatial: str = "IP",
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
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(
            n_basis=n_basis,
            partitioning=partitioning,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            should_apply_projection_back=should_apply_projection_back,
            should_record_loss=should_record_loss,
            reference_id=reference_id,
            rng=rng,
        )

        assert algorithm_spatial in algorithms_spatial, "Not support {}.".format(algorithms_spatial)
        assert 1 <= domain <= 2, "domain parameter should be chosen from [1, 2]."

        self.algorithm_spatial = algorithm_spatial
        self.domain = domain
        self.normalization = normalization

        if pair_selector is None and algorithm_spatial in ["IP2", "ISS2"]:
            self.pair_selector = functools.partial(sequential_pair_selector, sort=True)
        else:
            self.pair_selector = pair_selector

    def __repr__(self) -> str:
        s = "GaussILRMA("
        s += "n_basis={n_basis}"
        s += ", algorithm_spatial={algorithm_spatial}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalization={normalization}"
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

    def update_once(self) -> None:
        r"""Update NMF parameters and demixing filters once.
        """
        self.update_source_model()
        self.update_spatial_model()

        if self.normalization:
            self.normalize()

    def update_source_model(self) -> None:
        r"""Update NMF bases, activations, and latent variables.
        """

        if self.partitioning:
            self.update_latent()

        self.update_basis()
        self.update_activation()

    def update_latent(self) -> None:
        r"""Update latent variables in NMF.
        """
        p = self.domain

        if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
        else:
            Y = self.output

        Y2 = np.abs(Y) ** 2
        p2p = (p + 2) / p
        pp2 = p / (p + 2)

        Z = self.latent
        T, V = self.basis, self.activation

        TV = T[:, :, np.newaxis] * V[np.newaxis, :, :]
        ZTV = self.reconstruct_nmf(T, V, latent=Z)

        ZTVp2p = ZTV ** p2p
        TV_ZTVp2p = TV[np.newaxis, :, :, :] / ZTVp2p[:, :, np.newaxis, :]
        num = np.sum(TV_ZTVp2p * Y2[:, :, np.newaxis, :], axis=(1, 3))

        TV_ZTV = TV[np.newaxis, :, :, :] / ZTV[:, :, np.newaxis, :]
        denom = np.sum(TV_ZTV, axis=(1, 3))

        Z = ((num / denom) ** pp2) * Z
        Z = Z / Z.sum(axis=0)

        self.latent = Z

    def update_basis(self) -> None:
        r"""Update NMF bases.
        """
        p = self.domain

        if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
        else:
            Y = self.output

        Y2 = np.abs(Y) ** 2
        p2p = (p + 2) / p
        pp2 = p / (p + 2)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZV = Z[:, :, np.newaxis] * V[np.newaxis, :, :]
            ZTV = self.reconstruct_nmf(T, V, latent=Z)

            ZTVp2p = ZTV ** p2p
            ZV_ZTVp2p = ZV[:, np.newaxis, :, :] / ZTVp2p[:, :, np.newaxis, :]
            num = np.sum(ZV_ZTVp2p * Y2[:, :, np.newaxis, :], axis=(0, 3))

            ZV_ZTV = ZV[:, np.newaxis, :, :] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(ZV_ZTV, axis=(0, 3))
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)

            TVp2p = TV ** p2p
            V_TVp2p = V[:, np.newaxis, :, :] / TVp2p[:, :, np.newaxis, :]
            num = np.sum(V_TVp2p * Y2[:, :, np.newaxis, :], axis=3)

            V_TV = V[:, np.newaxis, :, :] / TV[:, :, np.newaxis, :]
            denom = np.sum(V_TV, axis=3)

        T = ((num / denom) ** pp2) * T
        T = self.flooring_fn(T)

        self.basis = T

    def update_activation(self):
        r"""Update NMF activations.
        """
        p = self.domain

        if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
        else:
            Y = self.output

        Y2 = np.abs(Y) ** 2
        p2p = (p + 2) / p
        pp2 = p / (p + 2)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            ZT = Z[:, np.newaxis, :] * T[np.newaxis, :, :]
            ZTV = self.reconstruct_nmf(T, V, latent=Z)

            ZTVp2p = ZTV ** p2p
            ZT_ZTVp2p = ZT[:, :, :, np.newaxis] / ZTVp2p[:, :, np.newaxis, :]
            num = np.sum(ZT_ZTVp2p * Y2[:, :, np.newaxis, :], axis=(0, 1))

            ZT_ZTV = ZT[:, :, :, np.newaxis] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(ZT_ZTV, axis=(0, 1))
        else:
            T, V = self.basis, self.activation

            TV = self.reconstruct_nmf(T, V)

            TVp2p = TV ** p2p
            T_TVp2p = T[:, :, :, np.newaxis] / TVp2p[:, :, np.newaxis, :]
            num = np.sum(T_TVp2p * Y2[:, :, np.newaxis, :], axis=1)

            T_TV = T[:, :, :, np.newaxis] / TV[:, :, np.newaxis, :]
            denom = np.sum(T_TV, axis=1)

        V = ((num / denom) ** pp2) * V
        V = self.flooring_fn(V)

        self.activation = V

    def update_spatial_model(self) -> None:
        r"""Update demixing filters once.

        If ``self.algorithm_spatial`` is ``"IP"`` or ``"IP1"``, ``update_once_ip1`` is called.
        If ``self.algorithm_spatial`` is ``"IP2"``, ``update_once_ip2`` is called.
        If ``self.algorithm_spatial`` is ``"ISS"`` or ``"ISS1"``, ``update_once_iss1`` is called.
        If ``self.algorithm_spatial`` is ``"ISS2"``, ``update_once_iss2`` is called.
        """
        if self.algorithm_spatial in ["IP", "IP1"]:
            self.update_spatial_model_ip1()
        elif self.algorithm_spatial in ["IP2"]:
            self.update_spatial_model_ip2()
        elif self.algorithm_spatial in ["ISS", "ISS1"]:
            self.update_spatial_model_iss1()
        elif self.algorithm_spatial in ["ISS2"]:
            self.update_spatial_model_iss2()
        else:
            raise NotImplementedError("Not support {}.".format(self.algorithm_spatial))

    def update_spatial_model_ip1(self) -> None:
        r"""Update demixing filters once using iterative projection.

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
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins

        p = self.domain
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

        E = np.eye(n_sources, n_channels)
        E = np.tile(E, reps=(n_bins, 1, 1))

        for src_idx in range(n_sources):
            w_n_Hermite = W[:, src_idx, :]
            U_n = U[:, src_idx, :, :]
            e_n = E[:, src_idx, :]

            WU = W @ U_n
            w_n = np.linalg.solve(WU, e_n)
            wUw = w_n[:, np.newaxis, :].conj() @ U_n @ w_n[:, :, np.newaxis]
            wUw = np.real(wUw[..., 0])
            wUw = np.maximum(wUw, 0)
            denom = np.sqrt(wUw)
            denom = self.flooring_fn(denom)
            w_n_Hermite = w_n.conj() / denom
            W[:, src_idx, :] = w_n_Hermite

        self.demix_filter = W

    def update_spatial_model_ip2(self) -> None:
        r"""Update demixing filters once using pairwise iterative projection.

        For :math:`m` and :math:`n` (:math:`m\neq n`),
        compute weighted covariance matrix as follows:

        .. math::
            \boldsymbol{V}_{im}^{(m,n)}
            &= \frac{1}{J}\sum_{j}\frac{1}{r_{ijm}} \
            \boldsymbol{y}_{ij}^{(m,n)}{\boldsymbol{y}_{ij}^{(m,n)}}^{\mathsf{H}} \\
            \boldsymbol{V}_{in}^{(m,n)}
            &= \frac{1}{J}\sum_{j}\frac{1}{r_{ijn}} \
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
            r_{ijn}
            = \sum_{k}z_{nk}t_{ik}v_{kj}

        if ``partitioning=True``.
        Otherwise,

        .. math::
            r_{ijn}
            = \sum_{k}t_{ikn}v_{kjn}.

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

        p = self.domain
        X, W = self.input, self.demix_filter

        E = np.eye(n_sources, n_channels)
        E = np.tile(E, reps=(n_bins, 1, 1))

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

        for m, n in self.pair_selector(n_sources):
            W_mn = W[:, (m, n), :]
            varphi_mn = varphi[(m, n), :, :]
            Y_mn = self.separate(X, demix_filter=W_mn)

            YY_Hermite = Y_mn[:, np.newaxis, :, :] * Y_mn[np.newaxis, :, :, :].conj()
            YY_Hermite = YY_Hermite.transpose(2, 0, 1, 3)  # (n_bins, 2, 2, n_frames)

            G_mn_YY = varphi_mn[:, :, np.newaxis, np.newaxis, :] * YY_Hermite
            U_mn = np.mean(G_mn_YY, axis=-1)  # (2, n_bins, 2, 2)

            U_m, U_n = U_mn
            _, H_mn = eigh(U_m, U_n)  # (n_bins, 2, 2)
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

    def update_spatial_model_iss1(self) -> None:
        n_sources = self.n_sources

        p = self.domain
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

        for src_idx in range(n_sources):
            Y_n = Y[src_idx]  # (n_bins, n_frames)

            YY_n_conj = Y * Y_n.conj()
            YY_n = np.abs(Y_n) ** 2
            num = np.mean(varphi * YY_n_conj, axis=-1)
            denom = np.mean(varphi * YY_n, axis=-1)
            denom = self.flooring_fn(denom)
            v_n = num / denom
            v_n[src_idx] = 1 - 1 / np.sqrt(denom[src_idx])

            Y = Y - v_n[:, :, np.newaxis] * Y_n

        self.output = Y

    def update_spatial_model_iss2(self) -> None:
        r"""Update estimated spectrograms once using pairwise iterative source steering.

        Then, we compute :math:`\boldsymbol{G}_{in}^{(m,m')}` \
        and :math:`\boldsymbol{f}_{in}^{(m,m')}` for :math:`m\neq m'`:

        .. math::
            \begin{array}{rclc}
                \boldsymbol{G}_{in}^{(m,m')}
                &=& {\displaystyle\frac{1}{J}\sum_{j}}\frac{1}{r_{ijn}}
                \boldsymbol{y}_{ij}^{(m,m')}{\boldsymbol{y}_{ij}^{(m,m')}}^{\mathsf{H}}
                &(n=1,\ldots,N), \\
                \boldsymbol{f}_{in}^{(m,m')}
                &=& {\displaystyle\frac{1}{J}\sum_{j}}
                \frac{1}{r_{ijn}}y_{ijn}^{*}\boldsymbol{y}_{ij}^{(m,m')}
                &(n\neq m,m'),
            \end{array}

        where

        .. math::
            r_{ijn}
            = \sum_{k}z_{nk}t_{ik}v_{kj}

        if ``partitioning=True``.
        Otherwise,

        .. math::
            r_{ijn}
            = \sum_{k}t_{ikn}v_{kjn}.

        Using :math:`\boldsymbol{G}_{in}^{(m,m')}` and :math:`\boldsymbol{f}_{in}`, \
        we compute

        .. math::
            \begin{array}{rclc}
                \boldsymbol{p}_{in}
                &=& \dfrac{\boldsymbol{h}_{in}}
                {\sqrt{\boldsymbol{h}_{in}^{\mathsf{H}}\boldsymbol{G}_{in}^{(m,m')}
                \boldsymbol{h}_{in}}} & (n=m,m'), \\
                \boldsymbol{q}_{in}
                &=& -{\boldsymbol{G}_{in}^{(m,m')}}^{-1}\boldsymbol{f}_{in}^{(m,m')}
                & (n\neq m,m'),
            \end{array}

        where :math:`\boldsymbol{h}_{in}` (:math:`n=m,m'`) is \
        a generalized eigenvector obtained from

        .. math::
            \boldsymbol{G}_{im}^{(m,m')}\boldsymbol{h}_{i}
            = \lambda_{i}\boldsymbol{G}_{im'}^{(m,m')}\boldsymbol{h}_{i}.

        Separated signal :math:`y_{ijn}` is updated as follows:

        .. math::
            y_{ijn}
            &\leftarrow\begin{cases}
            &\boldsymbol{p}_{in}^{\mathsf{H}}\boldsymbol{y}_{ij}^{(m,m')} & (n=m,m') \\
            &\boldsymbol{q}_{in}^{\mathsf{H}}\boldsymbol{y}_{ij}^{(m,m')} + y_{ijn} & (n\neq m,m')
            \end{cases}.

        """
        n_sources = self.n_sources

        p = self.domain
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

        for m, n in self.pair_selector(n_sources):
            # Split into main and sub
            Y_1, Y_m, Y_2, Y_n, Y_3 = np.split(Y, [m, m + 1, n, n + 1], axis=0)
            Y_main = np.concatenate([Y_m, Y_n], axis=0)  # (2, n_bins, n_frames)
            Y_sub = np.concatenate([Y_1, Y_2, Y_3], axis=0)  # (n_sources - 2, n_bins, n_frames)

            varphi_1, varphi_m, varphi_2, varphi_n, varphi_3 = np.split(
                varphi, [m, m + 1, n, n + 1], axis=0
            )
            varphi_main = np.concatenate([varphi_m, varphi_n], axis=0)
            varphi_sub = np.concatenate([varphi_1, varphi_2, varphi_3], axis=0)

            YY_main = Y_main[:, np.newaxis, :, :] * Y_main[np.newaxis, :, :, :].conj()
            YY_sub = Y_main[:, np.newaxis, :, :] * Y_sub[np.newaxis, :, :, :].conj()
            YY_main = YY_main.transpose(2, 0, 1, 3)
            YY_sub = YY_sub.transpose(1, 2, 0, 3)

            Y_main = Y_main.transpose(1, 0, 2)

            # Sub
            G_sub = np.mean(
                varphi_sub[:, :, np.newaxis, np.newaxis, :] * YY_main[np.newaxis, :, :, :, :],
                axis=-1,
            )
            F = np.mean(varphi_sub[:, :, np.newaxis, :] * YY_sub, axis=-1)
            Q = -np.linalg.inv(G_sub) @ F[:, :, :, np.newaxis]
            Q = Q.squeeze(axis=-1)
            Q = Q.transpose(1, 0, 2)
            QY = Q.conj() @ Y_main
            Y_sub = Y_sub + QY.transpose(1, 0, 2)

            # Main
            G_main = np.mean(
                varphi_main[:, :, np.newaxis, np.newaxis, :] * YY_main[np.newaxis, :, :, :, :],
                axis=-1,
            )
            G_m, G_n = G_main
            _, H_mn = eigh(G_m, G_n)
            h_mn = H_mn.transpose(2, 0, 1)
            hGh_mn = h_mn[:, :, np.newaxis, :].conj() @ G_main @ h_mn[:, :, :, np.newaxis]
            hGh_mn = np.squeeze(hGh_mn, axis=-1)
            hGh_mn = np.real(hGh_mn)
            hGh_mn = np.maximum(hGh_mn, 0)
            denom_mn = np.sqrt(hGh_mn)
            denom_mn = self.flooring_fn(denom_mn)
            P = h_mn / denom_mn
            P = P.transpose(1, 0, 2)
            Y_main = P.conj() @ Y_main
            Y_main = Y_main.transpose(1, 0, 2)

            # Concat
            Y_m, Y_n = np.split(Y_main, [1], axis=0)
            Y1, Y2, Y3 = np.split(Y_sub, [m, n - 1], axis=0)
            Y = np.concatenate([Y1, Y_m, Y2, Y_n, Y3], axis=0)

        self.output = Y

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

        if ``partitioning=False``, otherwise

        .. math::
            r_{ijn}
            = \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}.

        Returns:
            float:
                Computed loss.
        """
        p = self.domain

        if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
            Y2 = np.abs(Y) ** 2
        else:
            X, Y = self.input, self.output
            Y2 = np.abs(Y) ** 2
            X, Y = X.transpose(1, 0, 2), Y.transpose(1, 0, 2)
            X_Hermite = X.transpose(0, 2, 1).conj()
            XX_Hermite = X @ X_Hermite
            W = Y @ X_Hermite @ np.linalg.inv(XX_Hermite)

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
        loss = loss.sum(axis=0)

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


class TILRMA(ILRMAbase):
    r"""Independent low-rank matrix analysis (ILRMA) on student's-t distribution.

    Args:
        n_basis (int):
            Number of NMF bases.
        dof (float):
            Degree of freedom parameter in student's-t distribution.
        algorithm_spatial (str):
            Algorithm for demixing filter updates.
            Choose "IP", "IP1", "IP2", "ISS", "ISS1", or "ISS2".
            Default: "IP".
        domain (float):
            Domain parameter. Default: ``2``.
        partitioning (bool):
            Whether to use partioning function. Default: ``False``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        pair_selector (callable, optional):
            Selector to choose updaing pair in ``IP2`` and ``ISS2``.
            If ``None`` is given, ``partial(sequential_pair_selector, sort=True)`` is used.
            Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        normalization (bool or str, optional):
            Normalization of demixing filters and NMF parameters.
            Choose "power" or "projection_back".
            Default: ``"power"``.
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
        rng (numpy.random.Generator):
            Random number generator. This is mainly used to randomly initialize NMF.
            Default: ``numpy.random.default_rng()``.
    """

    def __init__(
        self,
        n_basis: int,
        dof: float,
        algorithm_spatial: str = "IP",
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
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(
            n_basis=n_basis,
            partitioning=partitioning,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            should_apply_projection_back=should_apply_projection_back,
            should_record_loss=should_record_loss,
            reference_id=reference_id,
            rng=rng,
        )

        assert algorithm_spatial in algorithms_spatial, "Not support {}.".format(algorithms_spatial)
        assert 1 <= domain <= 2, "domain parameter should be chosen from [1, 2]."

        self.dof = dof
        self.algorithm_spatial = algorithm_spatial
        self.domain = domain
        self.normalization = normalization

        if pair_selector is None and algorithm_spatial in ["IP2", "ISS2"]:
            self.pair_selector = functools.partial(sequential_pair_selector, sort=True)
        else:
            self.pair_selector = pair_selector

    def __repr__(self) -> str:
        s = "TILRMA("
        s += "n_basis={n_basis}"
        s += ", dof={dof}"
        s += ", algorithm_spatial={algorithm_spatial}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalization={normalization}"
        s += ", should_apply_projection_back={should_apply_projection_back}"
        s += ", should_record_loss={should_record_loss}"

        if self.should_apply_projection_back:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        r"""Update NMF parameters and demixing filters once.
        """
        self.update_source_model()
        self.update_spatial_model()

        if self.normalization:
            self.normalize()

    def update_source_model(self) -> None:
        r"""Update NMF bases, activations, and latent variables.
        """
        if self.partitioning:
            self.update_latent()

        self.update_basis()
        self.update_activation()

    def update_latent(self) -> None:
        r"""Update latent variables in NMF.
        """
        p = self.domain
        nu = self.dof

        if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
        else:
            Y = self.output

        Y2 = np.abs(Y) ** 2
        pp2 = p / (p + 2)
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

        Z = ((num / denom) ** pp2) * Z
        Z = Z / Z.sum(axis=0)

        self.latent = Z

    def update_basis(self) -> None:
        r"""Update NMF bases.
        """
        p = self.domain
        nu = self.dof

        if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
        else:
            Y = self.output

        Y2 = np.abs(Y) ** 2
        pp2 = p / (p + 2)
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

        T = ((num / denom) ** pp2) * T
        T = self.flooring_fn(T)

        self.basis = T

    def update_activation(self) -> None:
        r"""Update NMF activations.
        """
        p = self.domain
        nu = self.dof

        if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
        else:
            Y = self.output

        Y2 = np.abs(Y) ** 2
        pp2 = p / (p + 2)
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

        V = ((num / denom) ** pp2) * V
        V = self.flooring_fn(V)

        self.activation = V

    def update_spatial_model(self) -> None:
        r"""Update demixing filters once.

        If ``self.algorithm_spatial`` is ``"IP"`` or ``"IP1"``, ``update_once_ip1`` is called.
        If ``self.algorithm_spatial`` is ``"ISS"`` or ``"ISS1"``, ``update_once_iss1`` is called.
        """
        if self.algorithm_spatial in ["IP", "IP1"]:
            self.update_spatial_model_ip1()
        elif self.algorithm_spatial in ["ISS", "ISS1"]:
            self.update_spatial_model_iss1()
        else:
            raise NotImplementedError("Not support {}.".format(self.algorithm_spatial))

    def update_spatial_model_ip1(self) -> None:
        r"""Update demixing filters once using iterative projection.

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
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins

        p = self.domain
        nu = self.dof

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

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

        E = np.eye(n_sources, n_channels)
        E = np.tile(E, reps=(n_bins, 1, 1))

        for src_idx in range(n_sources):
            w_n_Hermite = W[:, src_idx, :]
            U_n = U[:, src_idx, :, :]
            e_n = E[:, src_idx, :]

            WU = W @ U_n
            w_n = np.linalg.solve(WU, e_n)
            wUw = w_n[:, np.newaxis, :].conj() @ U_n @ w_n[:, :, np.newaxis]
            wUw = np.real(wUw[..., 0])
            wUw = np.maximum(wUw, 0)
            denom = np.sqrt(wUw)
            denom = self.flooring_fn(denom)
            w_n_Hermite = w_n.conj() / denom
            W[:, src_idx, :] = w_n_Hermite

        self.demix_filter = W

    def update_spatial_model_iss1(self) -> None:
        n_sources = self.n_sources

        p = self.domain
        nu = self.dof

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

        for src_idx in range(n_sources):
            Y_n = Y[src_idx]  # (n_bins, n_frames)

            YY_n_conj = Y * Y_n.conj()
            YY_n = np.abs(Y_n) ** 2
            num = np.mean(varphi * YY_n_conj, axis=-1)
            denom = np.mean(varphi * YY_n, axis=-1)
            denom = self.flooring_fn(denom)
            v_n = num / denom
            v_n[src_idx] = 1 - 1 / np.sqrt(denom[src_idx])

            Y = Y - v_n[:, :, np.newaxis] * Y_n

        self.output = Y

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

        if ``partitioning=False``, otherwise

        .. math::
            r_{ijn}
            = \left(\sum_{k}t_{ikn}v_{kjn}\right)^{\frac{2}{p}}.

        Returns:
            float:
                Computed loss.
        """
        nu = self.dof
        p = self.domain

        if self.algorithm_spatial in ["IP", "IP1", "IP2"]:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
            Y2 = np.abs(Y) ** 2
        else:
            X, Y = self.input, self.output
            Y2 = np.abs(Y) ** 2
            X, Y = X.transpose(1, 0, 2), Y.transpose(1, 0, 2)
            X_Hermite = X.transpose(0, 2, 1).conj()
            XX_Hermite = X @ X_Hermite
            W = Y @ X_Hermite @ np.linalg.inv(XX_Hermite)

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
        loss = loss.sum(axis=0)

        return loss
