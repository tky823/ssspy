import functools
from typing import Callable, List, Optional, Union

import numpy as np

from ..linalg.mean import gmeanmh
from ..special.flooring import identity, max_flooring
from ._psd import to_psd
from .base import IterativeMethodBase

__all__ = ["GaussMNMF"]

EPS = 1e-10


class MNMFBase(IterativeMethodBase):
    r"""Base class of multichannel nonnegative matrix factorization (MNMF).

    Args:
        n_basis (int):
            Number of NMF bases.
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

    def _init_instant_covariance(self) -> None:
        r"""Initialize instantaneous covariance of input."""
        X = self.input
        XX = X[:, np.newaxis] * X[np.newaxis, :].conj()
        XX = XX.transpose(2, 3, 0, 1)
        self.instant_covariance = to_psd(XX, flooring_fn=self.flooring_fn)

    def _init_nmf(self, rng: Optional[np.random.Generator] = None) -> None:
        r"""Initialize NMF.

        Args:
            rng (numpy.random.Generator, optional):
                Random number generator. If ``None`` is given,
                ``np.random.default_rng()`` is used.
                Default: ``None``.
        """
        n_basis = self.n_basis
        n_sources = self.n_sources
        n_bins, n_frames = self.n_bins, self.n_frames

        if rng is None:
            rng = np.random.default_rng()

        if self.partitioning:
            if not hasattr(self, "basis"):
                T = rng.random((n_bins, n_basis))
                T = self.flooring_fn(T)
            else:
                # To avoid overwriting.
                T = self.basis.copy()

            if not hasattr(self, "activation"):
                V = rng.random((n_basis, n_frames))
                V = self.flooring_fn(V)
            else:
                # To avoid overwriting.
                V = self.activation.copy()

            if not hasattr(self, "latent"):
                Z = rng.random((n_sources, n_basis))
                Z = Z / Z.sum(axis=0)
                Z = self.flooring_fn(Z)
            else:
                # To avoid overwriting.
                Z = self.latent.copy()

            self.basis, self.activation = T, V
            self.latent = Z
        else:
            if not hasattr(self, "basis"):
                T = rng.random((n_sources, n_bins, n_basis))
                T = self.flooring_fn(T)
            else:
                # To avoid overwriting.
                T = self.basis.copy()

            if not hasattr(self, "activation"):
                V = rng.random((n_sources, n_basis, n_frames))
                V = self.flooring_fn(V)
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


class MNMF(MNMFbase):
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
            axis1 (int):
                First axis of covariance matrix. Default: ``-2``.
            axis2 (int):
                Second axis of covariance matrix. Default: ``-1``.

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


class FastMNMFbase(MNMFbase):
    def __init__(
        self,
        n_basis: int,
        n_sources: Optional[int] = None,
        partitioning: bool = False,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["FastMNMFbase"], None], List[Callable[["FastMNMFbase"], None]]]
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

    def _reset(self, **kwargs) -> None:
        super()._reset(**kwargs)

        n_channels = self.n_channels
        n_bins = self.n_bins

        if not hasattr(self, "diagonalizer"):
            Q = np.eye(n_channels, n_channels, dtype=np.complex128)
            Q = np.tile(Q, reps=(n_bins, 1, 1))
        else:
            if self.diagonalizer is None:
                Q = None
            else:
                # To avoid overwriting ``diagonalizer`` given by keyword arguments.
                Q = self.diagonalizer.copy()

        self.diagonalizer = Q


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

    def __repr__(self) -> str:
        s = "GaussMNMF("
        s += "n_basis={n_basis}"

        if self.n_sources is not None:
            s += ", n_sources={n_sources}"

        if hasattr(self, "n_channels"):
            s += ", n_channels={n_channels}"

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

    def update_once(self) -> None:
        r"""Update MNMF parameters once."""
        self.update_basis()
        self.update_activation()
        self.update_spatial()

        if self.normalization:
            # ensure unit trace of spatial property
            # before updates of latent variables in MNMF
            self.normalize(axis1=-2, axis2=-1)

        if self.partitioning:
            self.update_latent()

    def update_basis(self) -> None:
        r"""Update NMF bases by MM algorithm."""
        n_sources = self.n_sources
        n_frames = self.n_frames
        na = np.newaxis

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
            R = to_psd(R, flooring_fn=self.flooring_fn)

            trace_RXXRH, trace_RH = _compute_traces(XX, R, spatial=H)

            VRXXRH = np.sum(V[na, na, :] * trace_RXXRH[:, :, na], axis=-1)
            VRH = np.sum(V[na, na, :] * trace_RH[:, :, na], axis=-1)

            num = np.sum(Z[:, na, :] * VRXXRH, axis=0)
            denom = np.sum(Z[:, na, :] * VRH, axis=0)
        else:
            R = self.reconstruct_mnmf(T, V, H)
            R = to_psd(R, flooring_fn=self.flooring_fn)

            trace_RXXRH, trace_RH = _compute_traces(XX, R, spatial=H)

            num = np.sum(V[:, na, :, :] * trace_RXXRH[:, :, na, :], axis=-1)
            denom = np.sum(V[:, na, :, :] * trace_RH[:, :, na, :], axis=-1)

        T = T * np.sqrt(num / denom)
        T = self.flooring_fn(T)

        self.basis = T

    def update_activation(self) -> None:
        r"""Update NMF activations by MM algorithm."""
        n_sources = self.n_sources
        n_frames = self.n_frames
        na = np.newaxis

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
            R = to_psd(R, flooring_fn=self.flooring_fn)

            trace_RXXRH, trace_RH = _compute_traces(XX, R, spatial=H)

            TRXXRH = np.sum(T[na, :, :, na] * trace_RXXRH[:, :, na, :], axis=1)
            TRH = np.sum(T[na, :, :, na] * trace_RH[:, :, na, :], axis=1)

            num = np.sum(Z[:, :, na] * TRXXRH, axis=0)
            denom = np.sum(Z[:, :, na] * TRH, axis=0)
        else:
            R = self.reconstruct_mnmf(T, V, H)
            R = to_psd(R, flooring_fn=self.flooring_fn)

            trace_RXXRH, trace_RH = _compute_traces(XX, R, spatial=H)

            num = np.sum(T[:, :, :, na] * trace_RXXRH[:, :, na, :], axis=1)
            denom = np.sum(T[:, :, :, na] * trace_RH[:, :, na, :], axis=1)

        V = V * np.sqrt(num / denom)
        V = self.flooring_fn(V)

        self.activation = V

    def update_spatial(self) -> None:
        r"""Update spatial properties in NMF by MM algorithm."""
        na = np.newaxis

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
        R = to_psd(R, flooring_fn=self.flooring_fn)
        R_inverse = np.linalg.inv(R)
        RXXR = R_inverse @ XX @ R_inverse

        P = np.sum(Lamb[:, :, :, na, na] * R_inverse, axis=2)
        Q = np.sum(Lamb[:, :, :, na, na] * RXXR, axis=2)
        HQH = H @ Q @ H

        P = to_psd(P, flooring_fn=self.flooring_fn)
        HQH = to_psd(HQH, flooring_fn=self.flooring_fn)

        # geometric mean of P^(-1) and HQH
        H = gmeanmh(P, HQH, type=2)
        H = to_psd(H, flooring_fn=self.flooring_fn)

        self.spatial = H

    def update_latent(self) -> None:
        r"""Update latent variables in NMF by MM algorithm."""
        n_sources = self.n_sources
        n_frames = self.n_frames
        na = np.newaxis

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
        R = to_psd(R, flooring_fn=self.flooring_fn)

        trace_RXXRH, trace_RH = _compute_traces(XX, R, spatial=H)

        VRXXRH = np.sum(V[na, na, :] * trace_RXXRH[:, :, na], axis=-1)
        VRH = np.sum(V[na, na, :] * trace_RH[:, :, na], axis=-1)

        num = np.sum(T * VRXXRH, axis=1)
        denom = np.sum(T * VRH, axis=1)

        Z = Z * np.sqrt(num / denom)
        Z = Z / Z.sum(axis=0)

        self.latent = Z
