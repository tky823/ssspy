from typing import Optional, Union, List, Callable
import functools
import warnings

import numpy as np

from ._flooring import max_flooring
from ..algorithm import projection_back

__all__ = [
    "GaussILRMA",
]

algorithms_spatial = ["IP", "IP1", "IP2", "ISS", "ISS1", "ISS2"]
EPS = 1e-10


class ILRMAbase:
    r"""Base class of independent low-rank matrix analysis (ILRMA) [#kitamura2016determined]_.

    Args:
        n_basis (int):
            Number of NMF bases.
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

        self._init_nmf()

    def _init_nmf(self) -> None:
        r"""Initialize NMF.
        """
        n_basis = self.n_basis
        n_sources = self.n_sources
        n_bins, n_frames = self.n_bins, self.n_frames

        if self.partitioning:
            variance_latent = 1e-2  # TODO
            Z = np.random.rand(n_sources, n_basis) * variance_latent + 1 / n_sources
            Z = Z / Z.sum(axis=0)
            T = np.random.rand(n_bins, n_basis)
            V = np.random.rand(n_basis, n_frames)

            Z = self.flooring_fn(Z)
            T = self.flooring_fn(T)
            V = self.flooring_fn(V)

            self.latent = Z
            self.basis, self.activation = T, V
        else:
            T = np.random.rand(n_sources, n_bins, n_basis)
            V = np.random.rand(n_sources, n_basis, n_frames)

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

    def update_once(self) -> None:
        r"""Update demixing filters once.
        """
        raise NotImplementedError("Implement 'update_once' method.")

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
    def __init__(
        self,
        n_basis: int,
        algorithm_spatial: str = "IP",
        domain: float = 2,
        partitioning: bool = False,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["GaussILRMA"], None], List[Callable[["GaussILRMA"], None]]]
        ] = None,
        normalization: Optional[Union[str, bool]] = "projection_back",
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            n_basis=n_basis,
            partitioning=partitioning,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            should_apply_projection_back=should_apply_projection_back,
            should_record_loss=should_record_loss,
            reference_id=reference_id,
        )

        assert algorithm_spatial in algorithms_spatial, "Not support {}.".format(algorithms_spatial)
        assert 1 <= domain <= 2, "domain parameter should be chosen from [1, 2]."

        self.algorithm_spatial = algorithm_spatial
        self.domain = domain
        self.normalization = normalization

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
        r"""Update demixing filters once.

        If ``self.algorithm_spatial`` is ``"IP"`` or ``"IP1"``, ``update_once_ip1`` is called.
        """
        self.update_source_model()
        self.update_spatial_model()

        if self.normalization:
            self.normalize()

    def update_source_model(self) -> None:
        r"""Update NMF bases, activations, and latent variables.
        """
        p = self.domain
        X, W = self.input, self.demix_filter

        Y = self.separate(X, demix_filter=W)
        Y2 = np.abs(Y) ** 2
        p2p = (p + 2) / p
        pp2 = p / (p + 2)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            # Update basis
            ZV = Z[:, :, np.newaxis] * V[np.newaxis, :, :]
            ZTV = np.sum(ZV[:, np.newaxis, :, :] * T[np.newaxis, :, :, np.newaxis], axis=2)

            ZTVp2p = ZTV ** p2p
            ZV_ZTVp2p = ZV[:, np.newaxis, :, :] / ZTVp2p[:, :, np.newaxis, :]
            num = np.sum(ZV_ZTVp2p * Y2[:, :, np.newaxis, :], axis=(0, 3))

            ZV_ZTV = ZV[:, np.newaxis, :, :] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(ZV_ZTV, axis=(0, 3))

            T = ((num / denom) ** pp2) * T
            T = self.flooring_fn(T)

            # Update activation
            ZT = Z[:, np.newaxis, :] * T[np.newaxis, :, :]
            ZTV = np.sum(ZT[:, :, :, np.newaxis] * V[np.newaxis, np.newaxis, :, :], axis=2)

            ZTVp2p = ZTV ** p2p
            ZT_ZTVp2p = ZT[:, :, :, np.newaxis] / ZTVp2p[:, :, np.newaxis, :]
            num = np.sum(ZT_ZTVp2p * Y2[:, :, np.newaxis, :], axis=(0, 1))

            ZT_ZTV = ZT[:, :, :, np.newaxis] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(ZT_ZTV, axis=(0, 1))

            V = ((num / denom) ** pp2) * V
            V = self.flooring_fn(V)

            # Update latent
            TV = T[:, :, np.newaxis] * V[np.newaxis, :, :]
            ZTV = np.sum(Z[:, np.newaxis, :, np.newaxis] * TV[np.newaxis, :, :, :], axis=2)

            ZTVp2p = ZTV ** p2p
            TV_ZTVp2p = TV[np.newaxis, :, :, :] / ZTVp2p[:, :, np.newaxis, :]
            num = np.sum(TV_ZTVp2p * Y2[:, :, np.newaxis, :], axis=(1, 3))

            TV_ZTV = TV[np.newaxis, :, :, :] / ZTV[:, :, np.newaxis, :]
            denom = np.sum(TV_ZTV, axis=(1, 3))

            Z = ((num / denom) ** pp2) * Z
            Z = Z / Z.sum(axis=0)

            self.latent = Z
            self.basis, self.activation = T, V
        else:
            T, V = self.basis, self.activation

            # Update basis
            TV = T @ V

            TVp2p = TV ** p2p
            V_TVp2p = V[:, np.newaxis, :, :] / TVp2p[:, :, np.newaxis, :]
            num = np.sum(V_TVp2p * Y2[:, :, np.newaxis, :], axis=3)

            V_TV = V[:, np.newaxis, :, :] / TV[:, :, np.newaxis, :]
            denom = np.sum(V_TV, axis=3)

            T = ((num / denom) ** pp2) * T
            T = self.flooring_fn(T)

            # Update activation
            TV = T @ V

            TVp2p = TV ** p2p
            T_TVp2p = T[:, :, :, np.newaxis] / TVp2p[:, :, np.newaxis, :]
            num = np.sum(T_TVp2p * Y2[:, :, np.newaxis, :], axis=1)

            T_TV = T[:, :, :, np.newaxis] / TV[:, :, np.newaxis, :]
            denom = np.sum(T_TV, axis=1)

            V = ((num / denom) ** pp2) * V
            V = self.flooring_fn(V)

            self.basis, self.activation = T, V

    def update_spatial_model(self) -> None:
        r"""Update demixing filters once.
        """
        if self.algorithm_spatial in ["IP", "IP1"]:
            self.update_spatial_model_ip1()
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
            {\sqrt{\boldsymbol{w}_{in}^{\mathsf{H}}\boldsymbol{U}_{in}\boldsymbol{w}_{in}}}, \\

        where

        .. math::
            \boldsymbol{U}_{in}
            &= \frac{1}{J}\sum_{j}
            \frac{1}{\sum_{k}z_{nk}t_{ik}v_{kj}}
            \boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}

        if ``partitioning=True``, otherwise

        .. math::
            \boldsymbol{U}_{in}
            &= \frac{1}{J}\sum_{j}
            \frac{1}{\sum_{k}t_{ikn}v_{kjn}}
            \boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}.

        """
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins

        p = self.domain
        X, W = self.input, self.demix_filter

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation

            TV = T[:, :, np.newaxis] * V[np.newaxis, :, :]
            ZTV = np.sum(Z[:, np.newaxis, :, np.newaxis] * TV[np.newaxis, :, :, :], axis=2)
            ZTV2p = ZTV ** (2 / p)
            varphi = 1 / ZTV2p
        else:
            T, V = self.basis, self.activation

            TV2p = (T @ V) ** (2 / p)
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

    def normalize(self) -> None:
        r"""Normalize demixing filters and NMF parameters.
        """
        normalization = self.normalization

        assert normalization, "Set normalization."

        if self.partitioning:
            self.normalize_latent()
        else:
            self.normalize_wo_latent()

    def normalize_latent(self) -> None:
        r"""Normalize demixing filters and NMF parameters.

        Demixing filters and NMF parameters are normalized by

        .. math::
            \boldsymbol{w}_{in}
            &\leftarrow\frac{\boldsymbol{w}_{in}}{\psi_{in}},
            t_{ik}
            &\leftarrow t_{ik}\sum_{n}\frac{z_{nk}}{\psi_{in}^{p}},
            z_{nk}
            &\leftarrow \frac{\frac{z_{nk}}{\psi_{in}^{p}}}
            {\sum_{n'}\frac{z_{n'k}}{\psi_{in'}^{p}}},

        where :math:`\psi_{in}` is normalization term.
        :math:`0<p\leq 2` is a domain parameter.

        If self.normalization="power", \
        normalization term :math:`\psi_{in}` is computed as

        .. math::
            \psi_{in}
            = \sqrt{\frac{1}{IJ}|\boldsymbol{w}_{in}^{\mathsf{H}}
            \boldsymbol{x}_{ij}|^{2}}.

        """

        normalization = self.normalization

        p = self.domain
        X, W = self.input, self.demix_filter
        Z = self.latent
        T, V = self.basis, self.activation

        Y = self.separate(X, demix_filter=W)

        if type(normalization) is bool:
            # when normalization is True
            normalization = "power"

        if normalization == "power":
            Y2 = np.mean(np.abs(Y) ** 2, axis=(-2, -1))
            psi = np.sqrt(Y2)
            psi = self.flooring_fn(psi)
            W = W / psi[np.newaxis, :, np.newaxis]
            Z_psi = Z / (psi[:, np.newaxis] ** p)
            scale = np.sum(Z_psi, axis=0)
            T = T * scale[np.newaxis, :]
            Z = Z_psi / scale
        else:
            raise NotImplementedError("Normalization {} is not implemented.".format(normalization))

        self.demix_filter = W
        self.latent = Z
        self.basis, self.activation = T, V

    def normalize_wo_latent(self) -> None:
        r"""Normalize demixing filters and NMF parameters.

        Demixing filters and NMF parameters are normalized by

        .. math::
            \boldsymbol{w}_{in}
            &\leftarrow\frac{\boldsymbol{w}_{in}}{\psi_{in}},
            t_{ikn}
            &\leftarrow\frac{t_{ikn}}{\psi_{in}^{p}},

        where :math:`\psi_{in}` is normalization term.
        :math:`0<p\leq 2` is a domain parameter.

        If self.normalization="power", \
        normalization term :math:`\psi_{in}` is computed as

        .. math::
            \psi_{in}
            = \sqrt{\frac{1}{IJ}|\boldsymbol{w}_{in}^{\mathsf{H}}
            \boldsymbol{x}_{ij}|^{2}}.

        """
        normalization = self.normalization
        reference_id = self.reference_id

        p = self.domain
        X, W = self.input, self.demix_filter
        T, V = self.basis, self.activation

        Y = self.separate(X, demix_filter=W)

        if type(normalization) is bool:
            # when normalization is True
            normalization = "power"

        if normalization == "power":
            Y2 = np.mean(np.abs(Y) ** 2, axis=(-2, -1))  # (n_sources,)
            psi = np.sqrt(Y2)
            psi = self.flooring_fn(psi)
            W = W / psi[np.newaxis, :, np.newaxis]
            T = T / (psi[:, np.newaxis, np.newaxis] ** p)
        elif normalization == "projection_back":
            if reference_id is None:
                warnings.warn(
                    "channel 0 is used for reference_id \
                        of projection-back-based normalization.",
                    UserWarning,
                )
                reference_id = 0

            scale = np.linalg.inv(W)
            scale = scale[:, reference_id, :]
            W = W * scale[:, :, np.newaxis]
            scale = scale.transpose(1, 0)
            scale = np.abs(scale) ** p
            T = T * scale[:, :, np.newaxis]
        else:
            raise NotImplementedError("Normalization {} is not implemented.".format(normalization))

        self.demix_filter = W
        self.basis, self.activation = T, V

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L}
            = \frac{1}{J}\sum_{i,j}\left(\frac{|y_{ijn}|^{2}}{r_{ijn}}
            - \log\frac{|y_{ijn}|^{2}}{r_{ijn}}\right),

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
        X, W = self.input, self.demix_filter

        Y = self.separate(X, demix_filter=W)
        Y2 = np.abs(Y) ** 2

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation
            TV = T[:, :, np.newaxis] * V[np.newaxis, :, :]
            ZTV = np.sum(Z[:, np.newaxis, :, np.newaxis] * TV[np.newaxis, :, :, :], axis=2)
            ZTV2p = ZTV ** (2 / p)

            loss = Y2 / ZTV2p + (2 / p) * np.log(ZTV)
        else:
            T, V = self.basis, self.activation
            TV = T @ V
            TV2p = TV ** (2 / p)

            loss = Y2 / TV2p + (2 / p) * np.log(TV)

        logdet = self.compute_logdet(W)  # (n_bins,)

        loss = np.sum(loss.mean(axis=-1), axis=0) - 2 * logdet
        loss = loss.sum(axis=0)

        return loss
