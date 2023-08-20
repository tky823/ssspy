import functools
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

from ..linalg.mean import gmeanmh
from ..special.flooring import identity, max_flooring
from ..special.psd import to_psd
from ..utils.flooring import choose_flooring_fn
from ..utils.select_pair import sequential_pair_selector
from ._update_spatial_model import update_by_ip1, update_by_ip2
from .base import IterativeMethodBase

__all__ = ["GaussMNMF", "FastGaussMNMF"]

diagonalizer_algorithms = ["IP", "IP1", "IP2"]
EPS = 1e-10


class MNMFBase(IterativeMethodBase):
    r"""Base class of multichannel nonnegative matrix factorization (MNMF).

    Args:
        n_basis (int):
            Number of NMF bases.
        n_sources (int, optional):
            Number of sources to be separated.
            If ``None`` is given, ``n_sources`` is determined by number of channels
            in input spectrogram. Default: ``None``.
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
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel in multichannel Wiener filter. Default: ``0``.
        rng (numpy.random.Generator, optioinal):
            Random number generator. This is mainly used to randomly initialize PSDTF.
            If ``None`` is given, ``np.random.default_rng()`` is used.
            Default: ``None``.
    """

    def __init__(
        self,
        n_basis: int,
        n_sources: Optional[int] = None,
        partitioning: bool = False,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["MNMFBase"], None], List[Callable[["MNMFBase"], None]]]
        ] = None,
        normalization: bool = True,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(callbacks=callbacks, record_loss=record_loss)

        self.n_basis = n_basis
        self.n_sources = n_sources
        self.partitioning = partitioning

        if flooring_fn is None:
            self.flooring_fn = identity
        else:
            self.flooring_fn = flooring_fn

        self.normalization = normalization

        self.input = None
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

        self._reset(**kwargs)

        super().__call__(n_iter=n_iter, initial_call=initial_call)

        self.output = self.separate(self.input)

        return self.output

    def __repr__(self) -> str:
        s = "MNMF("
        s += "n_basis={n_basis}"

        if self.n_sources is not None:
            s += ", n_sources={n_sources}"

        if hasattr(self, "n_channels"):
            s += ", n_channels={n_channels}"

        s += ", partitioning={partitioning}"
        s += ", normalization={normalization}"
        s += ", record_loss={record_loss}"
        s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(self, **kwargs) -> None:
        r"""Reset attributes by given keyword arguments.

        Args:
            kwargs:
                Keyword arguments to set as attributes of MNMF.
        """
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        X = self.input

        n_sources = self.n_sources
        n_channels, n_bins, n_frames = X.shape

        if n_sources is None:
            n_sources = n_channels

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        self._init_instant_covariance()
        self._init_nmf(rng=self.rng)

        self.output = self.separate(X)

    def _init_instant_covariance(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Initialize instantaneous covariance of input.

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

        X = self.input
        XX = X[:, np.newaxis] * X[np.newaxis, :].conj()
        XX = XX.transpose(2, 3, 0, 1)
        self.instant_covariance = to_psd(XX, flooring_fn=flooring_fn)

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

            if not hasattr(self, "latent"):
                Z = rng.random((n_sources, n_basis))
                Z = Z / Z.sum(axis=0)
                Z = flooring_fn(Z)
            else:
                # To avoid overwriting.
                Z = self.latent.copy()

            self.basis, self.activation = T, V
            self.latent = Z
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

    def separate(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement 'separate' method.")

    def reconstruct_nmf(
        self,
        basis: np.ndarray,
        activation: np.ndarray,
        latent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Reconstruct single-channel NMF.

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
            numpy.ndarray of reconstructed single-channel NMF.
            The shape is (n_sources, n_bins, n_frames).
        """
        if latent is None:
            T, V = basis, activation
            Lamb = T @ V
        else:
            Z = latent
            T, V = basis, activation
            TV = T[:, :, np.newaxis] * V[np.newaxis, :, :]
            Lamb = np.sum(Z[:, np.newaxis, :, np.newaxis] * TV[np.newaxis, :, :, :], axis=2)

        return Lamb


class MNMF(MNMFBase):
    def __init__(
        self,
        n_basis: int,
        n_sources: Optional[int] = None,
        partitioning: bool = False,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[Union[Callable[["MNMF"], None], List[Callable[["MNMF"], None]]]] = None,
        normalization: bool = True,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            n_basis,
            n_sources=n_sources,
            partitioning=partitioning,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            normalization=normalization,
            record_loss=record_loss,
            reference_id=reference_id,
            rng=rng,
        )

    def _init_nmf(self, rng: Optional[np.random.Generator] = None) -> None:
        r"""Initialize NMF.

        Args:
            rng (numpy.random.Generator, optional):
                Random number generator. If ``None`` is given,
                ``np.random.default_rng()`` is used.
                Default: ``None``.
        """
        if rng is None:
            rng = np.random.default_rng()

        super()._init_nmf(rng=rng)

        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins

        if not hasattr(self, "spatial"):
            H = np.eye(n_channels, dtype=self.input.dtype)
            trace = np.trace(H, axis1=-2, axis2=-1)
            H = H / np.real(trace)
            H = np.tile(H, reps=(n_sources, n_bins, 1, 1))
        else:
            # To avoid overwriting.
            H = self.spatial.copy()

        self.spatial = H

    def reconstruct_mnmf(
        self,
        basis: np.ndarray,
        activation: np.ndarray,
        spatial: np.ndarray,
        latent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Reconstruct multichannel NMF.

        Args:
            basis (numpy.ndarray):
                Basis matrix with shape of (n_bins, n_basis).
            activation (numpy.ndarray):
                Activation matrix with shape of (n_basis, n_frames).
            spatial (numpy.ndarray):
                Spatial property with shape of (n_sources, n_bins, n_channels, n_channels).
            latent (numpy.ndarray, optional):
                Latent variables with shape of (n_sources, n_basis).

        Returns:
            numpy.ndarray of reconstructed multichannel NMF.
            The shape is (n_bins, n_frames, n_channels, n_channels).
        """
        T, V = basis, activation
        H = spatial

        if latent is None:
            Lamb = self.reconstruct_nmf(T, V)
        else:
            Lamb = self.reconstruct_nmf(T, V, latent=latent)

        R_n = Lamb[:, :, :, np.newaxis, np.newaxis] * H[:, :, np.newaxis, :, :]
        R = np.sum(R_n, axis=0)

        return R

    def normalize(self, axis1=-2, axis2=-1) -> None:
        r"""Ensure unit trace of spatial property of MNMF."""
        H = self.spatial
        n_dims = H.ndim

        axis1 = n_dims + axis1 if axis1 < 0 else axis1
        axis2 = n_dims + axis2 if axis2 < 0 else axis2

        assert axis1 == 2 and axis2 == 3

        trace = np.trace(H, axis1=axis1, axis2=axis2)
        trace = np.real(trace)
        H = H / trace[..., np.newaxis, np.newaxis]

        if self.partitioning:
            # When self.partitioning=True,
            # normalization may change value of cost function
            pass
        else:
            T = self.basis
            T = trace[:, :, np.newaxis] * T
            self.basis = T

        self.spatial = H


class FastMNMFBase(MNMFBase):
    r"""Base class of fast multichannel nonnegative matrix factorization (Fast MNMF).

    Args:
        n_basis (int):
            Number of NMF bases.
        n_sources (int, optional):
            Number of sources to be separated.
            If ``None`` is given, ``n_sources`` is determined by number of channels
            in input spectrogram. Default: ``None``.
        partitioning (bool):
            Whether to use partioning function. Default: ``False``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        normalization (bool or str):
            Normalization of diagonalizers and diagonal elements of spatial covariance matrices.
            Only power-based normalization is supported. Default: ``True``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel in multichannel Wiener filter. Default: ``0``.
        rng (numpy.random.Generator, optioinal):
            Random number generator. This is mainly used to randomly initialize PSDTF.
            If ``None`` is given, ``np.random.default_rng()`` is used.
            Default: ``None``.
    """

    def __init__(
        self,
        n_basis: int,
        n_sources: Optional[int] = None,
        partitioning: bool = False,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["FastMNMFBase"], None], List[Callable[["FastMNMFBase"], None]]]
        ] = None,
        normalization: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            n_basis,
            n_sources=n_sources,
            partitioning=partitioning,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            normalization=normalization,
            record_loss=record_loss,
            reference_id=reference_id,
            rng=rng,
        )

    def __repr__(self) -> str:
        s = "FastMNMF("
        s += "n_basis={n_basis}"

        if self.n_sources is not None:
            s += ", n_sources={n_sources}"

        if hasattr(self, "n_channels"):
            s += ", n_channels={n_channels}"

        s += ", partitioning={partitioning}"
        s += ", normalization={normalization}"
        s += ", record_loss={record_loss}"
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
                Keyword arguments to set as attributes of MNMF.
        """
        assert self.input is not None, "Specify data!"

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        X = self.input

        n_sources = self.n_sources
        n_channels, n_bins, n_frames = X.shape

        if n_sources is None:
            n_sources = n_channels

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        self._init_instant_covariance(flooring_fn=flooring_fn)
        self._init_nmf(flooring_fn=flooring_fn, rng=self.rng)
        self._init_diagonalizer(rng=self.rng)
        self._init_spatial(flooring_fn=flooring_fn, rng=self.rng)

        self.output = self.separate(X)

    def _init_diagonalizer(self, rng: Optional[np.random.Generator] = None) -> None:
        """Initialize diagonalizer.

        Args:
            rng (numpy.random.Generator, optional):
                Random number generator. If ``None`` is given,
                ``np.random.default_rng()`` is used.
                Default: ``None``.
        """
        n_channels = self.n_channels
        n_bins = self.n_bins

        if rng is None:
            rng = np.random.default_rng()

        if not hasattr(self, "diagonalizer"):
            Q = np.eye(n_channels, dtype=np.complex128)
            Q = np.tile(Q, reps=(n_bins, 1, 1))
        else:
            # To avoid overwriting.
            Q = self.diagonalizer.copy()

        self.diagonalizer = Q

    def _init_spatial(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initialize diagonal elements of spatial covariance matrices.

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
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if rng is None:
            rng = np.random.default_rng()

        if not hasattr(self, "spatial"):
            D = rng.random((n_bins, n_sources, n_channels))
            D = flooring_fn(D)
        else:
            D = self.spatial

        self.spatial = D

    def normalize(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Normalize diagonalizers and diagonal elements of spatial covariance matrices.

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
        else:
            raise NotImplementedError("Normalization {} is not implemented.".format(normalization))

    def normalize_by_power(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Normalize diagonalizers and diagonal elements of spatial covariance matrices by power.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Diagonalizers are normalized by

        .. math::
            \boldsymbol{q}_{im}
            \leftarrow\frac{\boldsymbol{q}_{im}}{\psi_{im}},

        where

        .. math::
            \psi_{im}
            = \sqrt{\frac{1}{IJ}\sum_{i,j}|\boldsymbol{q}_{im}^{\mathsf{H}}
            \boldsymbol{x}_{ij}|^{2}}.

        For diagonal elements of spatial covariance matrices,

        .. math::
            d_{inm}
            \leftarrow\frac{d_{inm}}{\psi_{im}^{2}}.
        """
        X = self.input
        Q, D = self.diagonalizer, self.spatial
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        QX = Q @ X.transpose(1, 0, 2)
        QX2 = np.mean(np.abs(QX) ** 2, axis=(0, 2))
        psi = np.sqrt(QX2)
        psi = flooring_fn(psi)

        Q = Q / psi[np.newaxis, :, np.newaxis]
        D = D / (psi**2)

        self.diagonalizer, self.spatial = Q, D


class GaussMNMF(MNMF):
    def __init__(
        self,
        n_basis: int,
        n_sources: Optional[int] = None,
        partitioning: bool = False,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["GaussMNMF"], None], List[Callable[["GaussMNMF"], None]]]
        ] = None,
        normalization: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            n_basis,
            n_sources=n_sources,
            partitioning=partitioning,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            normalization=normalization,
            record_loss=record_loss,
            reference_id=reference_id,
            rng=rng,
        )

    def __repr__(self) -> str:
        s = "GaussMNMF("
        s += "n_basis={n_basis}"

        if self.n_sources is not None:
            s += ", n_sources={n_sources}"

        if hasattr(self, "n_channels"):
            s += ", n_channels={n_channels}"

        s += ", partitioning={partitioning}"
        s += ", normalization={normalization}"
        s += ", record_loss={record_loss}"
        s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def separate(self, input: np.ndarray) -> np.ndarray:
        """Separate ``input`` using multichannel Wiener filter.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_sources, n_bins, n_frames).
        """
        n_sources = self.n_sources
        reference_id = self.reference_id

        X = input
        T, V = self.basis, self.activation
        H = self.spatial

        if self.partitioning:
            Lamb = self.reconstruct_nmf(T, V, latent=self.latent)
        else:
            Lamb = self.reconstruct_nmf(T, V)

        R_n = Lamb[:, :, :, np.newaxis, np.newaxis] * H[:, :, np.newaxis, :, :]
        R = np.sum(R_n, axis=0)
        R = to_psd(R, flooring_fn=self.flooring_fn)
        R = np.tile(R, reps=(n_sources, 1, 1, 1, 1))
        W_Hermite = np.linalg.solve(R, R_n)
        W = W_Hermite.transpose(0, 1, 2, 4, 3).conj()
        W_ref = W[:, :, :, reference_id, :]
        W_ref = W_ref.transpose(0, 3, 1, 2)
        Y = np.sum(W_ref * X, axis=1)

        return Y

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        Returns:
            Computed loss.
        """
        XX = self.instant_covariance
        T, V = self.basis, self.activation
        H = self.spatial

        if self.partitioning:
            R = self.reconstruct_mnmf(T, V, H, latent=self.latent)
        else:
            R = self.reconstruct_mnmf(T, V, H)

        R = to_psd(R, flooring_fn=self.flooring_fn)
        XXR_inv = np.linalg.solve(R, XX)  # Hermitian transpose of XX @ np.linalg.inv(R)
        trace = np.trace(XXR_inv, axis1=-2, axis2=-1)
        trace = np.real(trace)
        logdet = self.compute_logdet(R)
        loss = np.mean(trace + logdet, axis=-1)
        loss = loss.sum(axis=0)
        loss = loss.item()

        return loss

    def compute_logdet(self, reconstructed: np.ndarray) -> np.ndarray:
        r"""Compute log-determinant.

        Args:
            reconstructed:
                Reconstructed MNMF with shape of (\*, n_channels, n_channels).

        Returns:
            numpy.ndarray of computed log-determinant values.
            The shape is (\*).
        """
        _, logdet = np.linalg.slogdet(reconstructed)

        return logdet

    def update_once(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update MNMF parameters once.

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

        self.update_basis(flooring_fn=flooring_fn)
        self.update_activation(flooring_fn=flooring_fn)
        self.update_spatial(flooring_fn=flooring_fn)

        if self.normalization:
            # ensure unit trace of spatial property
            # before updates of latent variables in MNMF
            self.normalize(axis1=-2, axis2=-1)

        if self.partitioning:
            self.update_latent(flooring_fn=flooring_fn)

    def update_basis(
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

        """
        n_sources = self.n_sources
        n_frames = self.n_frames
        na = np.newaxis

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        def _compute_traces(
            target: np.ndarray, reconstructed: np.ndarray, spatial: np.ndarray
        ) -> np.ndarray:
            RXX = np.linalg.solve(reconstructed, target)
            R = np.tile(reconstructed, reps=(n_sources, 1, 1, 1, 1))
            H = np.tile(spatial[:, :, na, :, :], reps=(1, 1, n_frames, 1, 1))
            RH = np.linalg.solve(R, H)

            trace_RXXRH = np.trace(RXX @ RH, axis1=-2, axis2=-1)
            trace_RXXRH = np.real(trace_RXXRH)
            trace_RH = np.trace(RH, axis1=-2, axis2=-1)
            trace_RH = np.real(trace_RH)

            return trace_RXXRH, trace_RH

        XX = self.instant_covariance
        T, V = self.basis, self.activation
        H = self.spatial

        if self.partitioning:
            Z = self.latent
            R = self.reconstruct_mnmf(T, V, H, latent=Z)
            R = to_psd(R, flooring_fn=flooring_fn)

            trace_RXXRH, trace_RH = _compute_traces(XX, R, spatial=H)

            VRXXRH = np.sum(V[na, na, :] * trace_RXXRH[:, :, na], axis=-1)
            VRH = np.sum(V[na, na, :] * trace_RH[:, :, na], axis=-1)

            num = np.sum(Z[:, na, :] * VRXXRH, axis=0)
            denom = np.sum(Z[:, na, :] * VRH, axis=0)
        else:
            R = self.reconstruct_mnmf(T, V, H)
            R = to_psd(R, flooring_fn=flooring_fn)

            trace_RXXRH, trace_RH = _compute_traces(XX, R, spatial=H)

            num = np.sum(V[:, na, :, :] * trace_RXXRH[:, :, na, :], axis=-1)
            denom = np.sum(V[:, na, :, :] * trace_RH[:, :, na, :], axis=-1)

        T = T * np.sqrt(num / denom)
        T = flooring_fn(T)

        self.basis = T

    def update_activation(
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

        """
        n_sources = self.n_sources
        n_frames = self.n_frames
        na = np.newaxis

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        def _compute_traces(
            target: np.ndarray, reconstructed: np.ndarray, spatial: np.ndarray
        ) -> np.ndarray:
            RXX = np.linalg.solve(reconstructed, target)
            R = np.tile(reconstructed, reps=(n_sources, 1, 1, 1, 1))
            H = np.tile(spatial[:, :, na, :, :], reps=(1, 1, n_frames, 1, 1))
            RH = np.linalg.solve(R, H)

            trace_RXXRH = np.trace(RXX @ RH, axis1=-2, axis2=-1)
            trace_RXXRH = np.real(trace_RXXRH)
            trace_RH = np.trace(RH, axis1=-2, axis2=-1)
            trace_RH = np.real(trace_RH)

            return trace_RXXRH, trace_RH

        XX = self.instant_covariance
        T, V = self.basis, self.activation
        H = self.spatial

        if self.partitioning:
            Z = self.latent
            R = self.reconstruct_mnmf(T, V, H, latent=Z)
            R = to_psd(R, flooring_fn=flooring_fn)

            trace_RXXRH, trace_RH = _compute_traces(XX, R, spatial=H)

            TRXXRH = np.sum(T[na, :, :, na] * trace_RXXRH[:, :, na, :], axis=1)
            TRH = np.sum(T[na, :, :, na] * trace_RH[:, :, na, :], axis=1)

            num = np.sum(Z[:, :, na] * TRXXRH, axis=0)
            denom = np.sum(Z[:, :, na] * TRH, axis=0)
        else:
            R = self.reconstruct_mnmf(T, V, H)
            R = to_psd(R, flooring_fn=flooring_fn)

            trace_RXXRH, trace_RH = _compute_traces(XX, R, spatial=H)

            num = np.sum(T[:, :, :, na] * trace_RXXRH[:, :, na, :], axis=1)
            denom = np.sum(T[:, :, :, na] * trace_RH[:, :, na, :], axis=1)

        V = V * np.sqrt(num / denom)
        V = flooring_fn(V)

        self.activation = V

    def update_spatial(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update spatial properties in NMF by MM algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """
        na = np.newaxis
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        XX = self.instant_covariance
        T, V = self.basis, self.activation
        H = self.spatial

        if self.partitioning:
            Z = self.latent
            Lamb = self.reconstruct_nmf(T, V, latent=Z)
        else:
            Lamb = self.reconstruct_nmf(T, V)

        R_n = Lamb[:, :, :, na, na] * H[:, :, na, :, :]
        R = np.sum(R_n, axis=0)
        R = to_psd(R, flooring_fn=flooring_fn)
        R_inverse = np.linalg.inv(R)
        RXXR = R_inverse @ XX @ R_inverse

        P = np.sum(Lamb[:, :, :, na, na] * R_inverse, axis=2)
        Q = np.sum(Lamb[:, :, :, na, na] * RXXR, axis=2)
        HQH = H @ Q @ H

        P = to_psd(P, flooring_fn=flooring_fn)
        HQH = to_psd(HQH, flooring_fn=flooring_fn)

        # geometric mean of P^(-1) and HQH
        H = gmeanmh(P, HQH, type=2)
        H = to_psd(H, flooring_fn=flooring_fn)

        self.spatial = H

    def update_latent(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update latent variables in NMF by MM algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """
        n_sources = self.n_sources
        n_frames = self.n_frames
        na = np.newaxis

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        def _compute_traces(
            target: np.ndarray, reconstructed: np.ndarray, spatial: np.ndarray
        ) -> np.ndarray:
            RXX = np.linalg.solve(reconstructed, target)
            R = np.tile(reconstructed, reps=(n_sources, 1, 1, 1, 1))
            H = np.tile(spatial[:, :, na, :, :], reps=(1, 1, n_frames, 1, 1))
            RH = np.linalg.solve(R, H)

            trace_RXXRH = np.trace(RXX @ RH, axis1=-2, axis2=-1)
            trace_RXXRH = np.real(trace_RXXRH)
            trace_RH = np.trace(RH, axis1=-2, axis2=-1)
            trace_RH = np.real(trace_RH)

            return trace_RXXRH, trace_RH

        XX = self.instant_covariance
        T, V = self.basis, self.activation
        H, Z = self.spatial, self.latent

        R = self.reconstruct_mnmf(T, V, H, latent=Z)
        R = to_psd(R, flooring_fn=flooring_fn)

        trace_RXXRH, trace_RH = _compute_traces(XX, R, spatial=H)

        VRXXRH = np.sum(V[na, na, :] * trace_RXXRH[:, :, na], axis=-1)
        VRH = np.sum(V[na, na, :] * trace_RH[:, :, na], axis=-1)

        num = np.sum(T * VRXXRH, axis=1)
        denom = np.sum(T * VRH, axis=1)

        Z = Z * np.sqrt(num / denom)
        Z = Z / Z.sum(axis=0)

        self.latent = Z


class FastGaussMNMF(FastMNMFBase):
    r"""Fast multichannel nonnegative matrix factorization on Gaussian distribution \
    (Fast Gauss-MNMF).

    Args:
        n_basis (int):
            Number of NMF bases.
        n_sources (int, optional):
            Number of sources to be separated.
            If ``None`` is given, ``n_sources`` is determined by number of channels
            in input spectrogram. Default: ``None``.
        diagonalizer_algorithm (str):
            Algorithm for diagonalizers. Choose ``IP``, ``IP1``, or ``IP2``.
            Default: ``IP``.
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
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel in multichannel Wiener filter. Default: ``0``.
        rng (numpy.random.Generator, optioinal):
            Random number generator. This is mainly used to randomly initialize PSDTF.
            If ``None`` is given, ``np.random.default_rng()`` is used.
            Default: ``None``.
    """

    def __init__(
        self,
        n_basis: int,
        n_sources: Optional[int] = None,
        diagonalizer_algorithm: str = "IP",
        partitioning: bool = False,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        pair_selector: Optional[Callable[[int], Iterable[Tuple[int, int]]]] = None,
        callbacks: Optional[
            Union[Callable[["FastGaussMNMF"], None], List[Callable[["FastGaussMNMF"], None]]]
        ] = None,
        normalization: bool = True,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            n_basis,
            n_sources=n_sources,
            partitioning=partitioning,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            normalization=normalization,
            record_loss=record_loss,
            reference_id=reference_id,
            rng=rng,
        )

        assert diagonalizer_algorithm in diagonalizer_algorithms, "Not support {}.".format(
            diagonalizer_algorithm
        )
        assert not partitioning, "partitioning function is not supported."

        self.diagonalizer_algorithm = diagonalizer_algorithm

        if pair_selector is None:
            if diagonalizer_algorithm == "IP2":
                self.pair_selector = sequential_pair_selector
        else:
            self.pair_selector = pair_selector

    def __repr__(self) -> str:
        s = "FastGaussMNMF("
        s += "n_basis={n_basis}"

        if self.n_sources is not None:
            s += ", n_sources={n_sources}"

        if hasattr(self, "n_channels"):
            s += ", n_channels={n_channels}"

        s += ", diagonalizer_algorithm={diagonalizer_algorithm}"
        s += ", partitioning={partitioning}"
        s += ", record_loss={record_loss}"
        s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def separate(self, input: np.ndarray) -> np.ndarray:
        """Separate ``input`` using multichannel Wiener filter.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_sources, n_bins, n_frames).
        """
        na = np.newaxis
        n_sources = self.n_sources
        reference_id = self.reference_id

        X = input
        T, V = self.basis, self.activation
        Q, D = self.diagonalizer, self.spatial

        if self.partitioning:
            Lamb = self.reconstruct_nmf(T, V, latent=self.latent)
        else:
            Lamb = self.reconstruct_nmf(T, V)

        D = D.transpose(1, 0, 2)

        Q_inverse = np.linalg.inv(Q)
        Q_inverse_Hermite = Q_inverse.transpose(0, 2, 1).conj()
        QQ_Hermite = Q_inverse[:, :, :, na] * Q_inverse_Hermite[:, na, :, :]

        LambD = Lamb[:, :, :, na] * D[:, :, na, :]

        R_n = np.sum(LambD[:, :, :, na, :, na] * QQ_Hermite[:, na, :, :, :], axis=4)
        R = np.sum(R_n, axis=0)
        R = to_psd(R, flooring_fn=self.flooring_fn)
        R = np.tile(R, reps=(n_sources, 1, 1, 1, 1))
        W_Hermite = np.linalg.solve(R, R_n)
        W = W_Hermite.transpose(0, 1, 2, 4, 3).conj()
        W_ref = W[:, :, :, reference_id, :]
        W_ref = W_ref.transpose(0, 3, 1, 2)
        Y = np.sum(W_ref * X, axis=1)

        return Y

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is defined as follows:

        .. math::
            \mathcal{L}
            &:=-\frac{1}{J}\sum_{i,j}\left\{
            \mathrm{tr}\left(
            \boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}\boldsymbol{R}_{ij}^{-1}
            \right)
            - \log\det\boldsymbol{R}_{ij}
            \right\} \\
            &:=\frac{1}{J}\sum_{i,j,m}\left\{
            \frac{|\boldsymbol{q}_{im}^{\mathsf{H}}\boldsymbol{x}_{ij}|^{2}}
            {\sum_{n}\lambda_{ijn}d_{inm}}
            + \log\sum_{n}\lambda_{ijn}d_{inm}\right\}
            - 2\sum_{i}\log|\det\boldsymbol{Q}_{i}|.

        Returns:
            Computed loss.
        """
        X = self.input
        T, V = self.basis, self.activation
        Q, D = self.diagonalizer, self.spatial
        na = np.newaxis

        if self.partitioning:
            Lamb = self.reconstruct_nmf(T, V, latent=self.latent)
        else:
            Lamb = self.reconstruct_nmf(T, V)

        D = D.transpose(1, 0, 2)
        LambD = np.sum(Lamb[:, :, na, :] * D[:, :, :, na], axis=0)
        QX = Q @ X.transpose(1, 0, 2)
        QX2 = np.abs(QX) ** 2
        logdetQ = self.compute_logdet(Q)
        loss = np.sum(QX2 / LambD + np.log(LambD), axis=1)
        loss = np.mean(loss, axis=-1) - 2 * logdetQ
        loss = loss.sum(axis=0)
        loss = loss.item()

        return loss

    def compute_logdet(self, diagonalizer: np.ndarray) -> np.ndarray:
        r"""Compute log-determinant.

        Args:
            reconstructed:
                Diagonalizer with shape of (\*, n_channels, n_channels).

        Returns:
            numpy.ndarray of computed log-determinant values.
            The shape is (\*).
        """
        _, logdet = np.linalg.slogdet(diagonalizer)

        return logdet

    def update_once(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update MNMF parameters, diagonalizers, and diagonal elements of \
        spatial covariance matrices once.

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

        self.update_basis(flooring_fn=flooring_fn)
        self.update_activation(flooring_fn=flooring_fn)
        self.update_diagonalizer(flooring_fn=flooring_fn)
        self.update_spatial()

        if self.normalization:
            self.normalize(flooring_fn=flooring_fn)

    def update_basis(
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
            t_{ikn}
            \leftarrow\left[
            \frac{\displaystyle\sum_{j,m}\frac{|\boldsymbol{q}_{im}^{\mathsf{H}}\boldsymbol{x}_{ij}|^{2}d_{inm}v_{kjn}}
            {\left(\sum_{k',n'}t_{ik'n'}v_{k'jn'}d_{in'm}\right)^{2}}}
            {\displaystyle\sum_{j,m}\dfrac{d_{inm}v_{kjn}}{\sum_{k',n'}t_{ik'n'}v_{k'jn'}d_{in'm}}}
            \right]^{\frac{1}{2}}t_{ikn}.
        """
        assert not self.partitioning, "partitioning function is not supported."

        na = np.newaxis
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        X = self.input
        T, V = self.basis, self.activation
        Q, D = self.diagonalizer, self.spatial

        if self.partitioning:
            Lamb = self.reconstruct_nmf(T, V, latent=self.latent)
        else:
            Lamb = self.reconstruct_nmf(T, V)

        D = D.transpose(1, 0, 2)
        LambD = Lamb[:, :, :, na] * D[:, :, na, :]
        LambD = np.sum(LambD, axis=0)
        QX = Q @ X.transpose(1, 0, 2)
        QX = np.abs(QX)
        QX = QX.transpose(0, 2, 1)
        QXLambD = (QX / LambD) ** 2
        DQXLambD = np.sum(D[:, :, na, :] * QXLambD, axis=-1)
        DLambD = np.sum(D[:, :, na, :] / LambD, axis=-1)

        num = np.sum(V[:, na, :] * DQXLambD[:, :, na], axis=-1)
        denom = np.sum(V[:, na, :] * DLambD[:, :, na], axis=-1)

        T = T * np.sqrt(num / denom)
        T = flooring_fn(T)

        self.basis = T

    def update_activation(
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
            v_{kjn}
            \leftarrow\left[
            \frac{\displaystyle\sum_{i,m}\frac{|\boldsymbol{q}_{im}^{\mathsf{H}}\boldsymbol{x}_{ij}|^{2}d_{inm}t_{ikn}}
            {\left(\sum_{k',n'}t_{ik'n'}v_{k'jn'}d_{in'm}\right)^{2}}}
            {\displaystyle\sum_{i,m}\dfrac{d_{inm}t_{ikn}}{\sum_{k',n'}t_{ik'n'}v_{k'jn'}d_{in'm}}}
            \right]^{\frac{1}{2}}v_{kjn}.
        """
        assert not self.partitioning, "partitioning function is not supported."

        na = np.newaxis
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        X = self.input
        T, V = self.basis, self.activation
        Q, D = self.diagonalizer, self.spatial

        if self.partitioning:
            Lamb = self.reconstruct_nmf(T, V, latent=self.latent)
        else:
            Lamb = self.reconstruct_nmf(T, V)

        D = D.transpose(1, 0, 2)
        LambD = Lamb[:, :, :, na] * D[:, :, na, :]
        LambD = np.sum(LambD, axis=0)
        QX = Q @ X.transpose(1, 0, 2)
        QX = np.abs(QX)
        QX = QX.transpose(0, 2, 1)
        QXLambD = (QX / LambD) ** 2
        DQXLambD = np.sum(D[:, :, na, :] * QXLambD, axis=-1)
        DLambD = np.sum(D[:, :, na, :] / LambD, axis=-1)

        num = np.sum(T[:, :, :, na] * DQXLambD[:, :, na, :], axis=1)
        denom = np.sum(T[:, :, :, na] * DLambD[:, :, na, :], axis=1)

        V = V * np.sqrt(num / denom)
        V = flooring_fn(V)

        self.activation = V

    def update_diagonalizer(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        """Update diagonalizer.

        - If ``diagonalizer_algorithm`` is ``IP`` or ``IP1``, \
            ``update_diagonalizer_model_ip1`` is called.
        - If ``diagonalizer_algorithm`` is ``IP2``, \
            ``update_diagonalizer_model_ip2`` is called.

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

        if self.diagonalizer_algorithm in ["IP", "IP1"]:
            self.update_diagonalizer_ip1(flooring_fn=flooring_fn)
        elif self.diagonalizer_algorithm in ["IP2"]:
            self.update_diagonalizer_ip2(flooring_fn=flooring_fn)
        else:
            raise NotImplementedError("Not support {}.".format(self.diagonalizer_algorithm))

    def update_diagonalizer_ip1(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update diagonalizer once using iterative projection.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        Diagonalizers are updated sequentially for :math:`m=1,\ldots,M` as follows:

        .. math::
            \boldsymbol{q}_{im}
            &\leftarrow\left(\boldsymbol{Q}_{im}^{\mathsf{H}}\boldsymbol{U}_{im}\right)^{-1} \
            \boldsymbol{e}_{m}, \\
            \boldsymbol{q}_{im}
            &\leftarrow\frac{\boldsymbol{q}_{im}}
            {\sqrt{\boldsymbol{q}_{im}^{\mathsf{H}}\boldsymbol{U}_{im}\boldsymbol{q}_{im}}},

        where

        .. math::
            \boldsymbol{U}_{im}
            = \frac{1}{J}\sum_{j}
            \frac{\boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}}
            {\sum_{n}\left(\sum_{k}z_{nk}t_{ik}v_{kj}\right)d_{inm}}

        if ``partitioning=True``, otherwise

        .. math::
            \boldsymbol{U}_{im}
            = \frac{1}{J}\sum_{j}
            \frac{\boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}}
            {\sum_{n}\left(\sum_{k}t_{ikn}v_{kjn}\right)d_{inm}}.
        """
        assert not self.partitioning, "partitioning function is not supported."

        na = np.newaxis
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        X = self.input
        T, V = self.basis, self.activation
        Q, D = self.diagonalizer, self.spatial

        if self.partitioning:
            Lamb = self.reconstruct_nmf(T, V, latent=self.latent)
        else:
            Lamb = self.reconstruct_nmf(T, V)

        XX_Hermite = X[:, na, :, :] * X[na, :, :, :].conj()
        XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)

        Lamb = Lamb.transpose(1, 0, 2)
        LambD = np.sum(Lamb[:, :, na, :] * D[:, :, :, na], axis=1)
        varphi = 1 / LambD

        varphi_XX = varphi[:, :, na, na, :] * XX_Hermite[:, na, :, :, :]
        U = np.mean(varphi_XX, axis=-1)

        self.diagonalizer = update_by_ip1(Q, U, flooring_fn=flooring_fn)

    def update_diagonalizer_ip2(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update diagonalizer once using pairwise iterative projection.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        For :math:`m_{1}` and :math:`m_{2}` (:math:`m_{1}\neq m_{2}`),
        compute weighted covariance matrix as follows:

        .. math::
            \boldsymbol{U}_{im}
            = \frac{1}{J}\sum_{j}
            \frac{\boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}}{\sum_{n}\lambda_{ijn}d_{inm}},

        :math:`\lambda_{ijn}` is computed by

        .. math::
            \lambda_{ijn}=\sum_{k}z_{nk}t_{ik}v_{kj}

        if ``partitioning=True``.
        Otherwise,

        .. math::
            \lambda_{ijn}=\sum_{k}t_{ikn}v_{kjn}.

        Using :math:`\boldsymbol{U}_{im_{1}}` and
        :math:`\boldsymbol{U}_{im_{2}}`, we compute generalized eigenvectors.

        .. math::
            \left({\boldsymbol{P}_{im_{1}}^{(m_{1},m_{2})}}^{\mathsf{H}}\boldsymbol{U}_{im_{1}}
            \boldsymbol{P}_{im_{1}}^{(m_{1},m_{2})}\right)\boldsymbol{h}_{i}
            = \mu_{i}
            \left({\boldsymbol{P}_{im_{2}}^{(m_{1},m_{2})}}^{\mathsf{H}}\boldsymbol{U}_{im_{2}}
            \boldsymbol{P}_{im_{2}}^{(m_{1},m_{2})}\right)\boldsymbol{h}_{i},

        where

        .. math::
            \boldsymbol{P}_{im_{1}}^{(m_{1},m_{2})}
            &= (\boldsymbol{Q}_{i}\boldsymbol{U}_{im_{1}})^{-1}
            (
            \begin{array}{cc}
                \boldsymbol{e}_{m_{1}} & \boldsymbol{e}_{m_{2}}
            \end{array}
            ), \\
            \boldsymbol{P}_{im_{2}}^{(m_{1},m_{2})}
            &= (\boldsymbol{Q}_{i}\boldsymbol{U}_{im_{2}})^{-1}
            (
            \begin{array}{cc}
                \boldsymbol{e}_{m_{1}} & \boldsymbol{e}_{m_{2}}
            \end{array}
            ).

        After that, we standardize two eigenvectors :math:`\boldsymbol{h}_{im_{1}}`
        and :math:`\boldsymbol{h}_{im_{2}}`.

        .. math::
            \boldsymbol{h}_{im_{1}}
            &\leftarrow\frac{\boldsymbol{h}_{im_{1}}}
            {\sqrt{\boldsymbol{h}_{im_{1}}^{\mathsf{H}}
            \left({\boldsymbol{P}_{im_{1}}^{(m_{1},m_{2})}}^{\mathsf{H}}\boldsymbol{U}_{im_{1}}
            \boldsymbol{P}_{im_{1}}^{(m_{1},m_{2})}\right)
            \boldsymbol{h}_{im_{1}}}}, \\
            \boldsymbol{h}_{im_{2}}
            &\leftarrow\frac{\boldsymbol{h}_{im_{2}}}
            {\sqrt{\boldsymbol{h}_{im_{2}}^{\mathsf{H}}
            \left({\boldsymbol{P}_{im_{2}}^{(m_{1},m_{2})}}^{\mathsf{H}}\boldsymbol{U}_{im_{2}}
            \boldsymbol{P}_{im_{2}}^{(m_{1},m_{2})}\right)
            \boldsymbol{h}_{im_{2}}}}.

        Then, update :math:`\boldsymbol{q}_{im_{1}}` and :math:`\boldsymbol{q}_{im_{2}}`
        simultaneously.

        .. math::
            \boldsymbol{q}_{im_{1}}
            &\leftarrow \boldsymbol{P}_{im_{1}}^{(m_{1},m_{2})}\boldsymbol{h}_{im_{1}} \\
            \boldsymbol{q}_{im_{2}}
            &\leftarrow \boldsymbol{P}_{im_{2}}^{(m_{1},m_{2})}\boldsymbol{h}_{im_{2}}

        At each iteration, we update pairs of :math:`m_{1}` and :math:`m_{2}`
        for :math:`m_{1}\neq m_{2}`.
        """
        assert not self.partitioning, "partitioning function is not supported."

        na = np.newaxis
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        X = self.input
        T, V = self.basis, self.activation
        Q, D = self.diagonalizer, self.spatial

        if self.partitioning:
            Lamb = self.reconstruct_nmf(T, V, latent=self.latent)
        else:
            Lamb = self.reconstruct_nmf(T, V)

        XX_Hermite = X[:, na, :, :] * X[na, :, :, :].conj()
        XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)

        Lamb = Lamb.transpose(1, 0, 2)
        LambD = np.sum(Lamb[:, :, na, :] * D[:, :, :, na], axis=1)
        varphi = 1 / LambD

        varphi_XX = varphi[:, :, na, na, :] * XX_Hermite[:, na, :, :, :]
        U = np.mean(varphi_XX, axis=-1)

        self.diagonalizer = update_by_ip2(
            Q, U, flooring_fn=flooring_fn, pair_selector=self.pair_selector
        )

    def update_spatial(self) -> None:
        r"""Update diagonal elements of spatial covariance matrix by MM algorithm.

        Update :math:`d_{inm}` as follows:

        .. math::
            d_{inm}\leftarrow\left[
            \dfrac{\displaystyle\sum_{j}\frac{\lambda_{ijn}|\boldsymbol{q}_{im}^{\mathsf{H}}\boldsymbol{x}_{ij}|^{2}}
            {\left(\sum_{n'}\lambda_{ijn'}d_{in'm}\right)^{2}}}
            {\displaystyle\sum_{j}\frac{\lambda_{ijn}}{\sum_{n'}\lambda_{ijn'}d_{in'm}}}
            \right]^{\frac{1}{2}}d_{inm}.
        """
        assert not self.partitioning, "partitioning function is not supported."

        na = np.newaxis

        X = self.input
        T, V = self.basis, self.activation
        Q, D = self.diagonalizer, self.spatial

        if self.partitioning:
            Lamb = self.reconstruct_nmf(T, V, latent=self.latent)
        else:
            Lamb = self.reconstruct_nmf(T, V)

        QX = Q @ X.transpose(1, 0, 2)
        QX = np.abs(QX)
        QX2 = QX**2

        Lamb = Lamb.transpose(1, 0, 2)
        LambD = np.sum(Lamb[:, :, na, :] * D[:, :, :, na], axis=1)
        LambD2 = LambD**2
        Lamb_LambD2 = Lamb[:, :, na] / LambD2[:, na, :]
        num = np.sum(Lamb_LambD2 * QX2[:, na, :, :], axis=-1)

        Lamb_LambD = Lamb[:, :, na] / LambD[:, na, :]
        denom = np.sum(Lamb_LambD, axis=-1)

        D = np.sqrt(num / denom) * D

        self.spatial = D
