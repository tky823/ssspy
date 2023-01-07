import functools
from typing import Callable, List, Optional, Union

import numpy as np

from ._flooring import identity, max_flooring
from ._psd import to_psd
from .base import IterativeMethodBase

__all__ = ["GaussMNMF"]

EPS = 1e-10


class MNMFbase(IterativeMethodBase):
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
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["MNMFbase"], None], List[Callable[["MNMFbase"], None]]]
        ] = None,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(callbacks=callbacks, record_loss=record_loss)

        self.n_basis = n_basis
        self.n_sources = n_sources

        if flooring_fn is None:
            self.flooring_fn = identity
        else:
            self.flooring_fn = flooring_fn

        self.input = None
        self.reference_id = reference_id

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

    def __call__(self, input: np.ndarray, n_iter: int = 100, **kwargs) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                The number of iterations of demixing filter updates.
                Default: ``100``.

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_channels, n_bins, n_frames).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        super().__call__(n_iter=n_iter)

        self.output = self.separate(self.input)

        return self.output

    def __repr__(self) -> str:
        s = "MNMF("
        s += "n_basis={n_basis}"

        if self.n_sources is not None:
            s += ", n_sources={n_sources}"

        if hasattr(self, "n_channels"):
            s += ", n_channels={n_channels}"

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

        self._init_nmf(rng=self.rng)

        self.output = self.separate(X)

    def _init_nmf(self, rng: Optional[np.random.Generator] = None) -> None:
        r"""Initialize NMF.

        Args:
            rng (numpy.random.Generator, optional):
                Random number generator. If ``None`` is given,
                ``np.random.default_rng()`` is used.
                Default: ``None``.
        """
        n_basis = self.n_basis
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins, n_frames = self.n_bins, self.n_frames

        if rng is None:
            rng = np.random.default_rng()

        if not hasattr(self, "latent"):
            Z = rng.random((n_sources, n_basis))
            Z = Z / Z.sum(axis=0)
            Z = self.flooring_fn(Z)
        else:
            # To avoid overwriting.
            Z = self.latent.copy()

        if not hasattr(self, "spatial"):
            H = np.eye(n_channels)
            H = np.tile(H, reps=(n_sources, n_bins, 1, 1))
        else:
            # To avoid overwriting.
            H = self.spatial.copy()

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

        self.latent = Z
        self.spatial = H
        self.basis, self.activation = T, V

    def separate(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement 'separate' method.")

    def reconstruct_nmf(
        self,
        basis: np.ndarray,
        activation: np.ndarray,
        latent: np.ndarray,
    ) -> np.ndarray:
        r"""Reconstruct single-channel NMF.

        Args:
            basis (numpy.ndarray):
                Basis matrix with shape of (n_bins, n_basis).
            activation (numpy.ndarray):
                Activation matrix with shape of (n_basis, n_frames).
            latent (numpy.ndarray):
                Latent variables with shape of (n_sources, n_basis).
            axis1 (int):
                First axis of covariance matrix. Default: ``-2``.
            axis2 (int):
                Second axis of covariance matrix. Default: ``-1``.

        Returns:
            numpy.ndarray of reconstructed single-channel NMF.
            The shape is (n_sources, n_bins, n_frames).
        """
        T, V = basis, activation
        Z = latent

        TV = T[:, :, np.newaxis] * V[np.newaxis, :, :]
        ZTV = np.sum(Z[:, np.newaxis, :, np.newaxis] * TV[np.newaxis, :, :, :], axis=2)

        return ZTV

    def reconstruct_mnmf(
        self,
        basis: np.ndarray,
        activation: np.ndarray,
        spatial: np.ndarray,
        latent: np.ndarray,
    ) -> np.ndarray:
        r"""Reconstruct multichannel NMF.

        Args:
            basis (numpy.ndarray):
                Basis matrix with shape of (n_bins, n_basis).
            activation (numpy.ndarray):
                Activation matrix with shape of (n_basis, n_frames).
            spatial (numpy.ndarray):
                Spatial property with shape of (n_sources, n_bins, n_channels, n_channels).
            latent (numpy.ndarray):
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
        H, Z = spatial, latent

        ZTV = self.reconstruct_nmf(T, V, Z)
        R_hat_n = ZTV[:, :, :, np.newaxis, np.newaxis] * H[:, :, np.newaxis, :, :]
        R_hat = np.sum(R_hat_n, axis=0)

        return R_hat


class GaussMNMF(MNMFbase):
    def __init__(
        self,
        n_basis: int,
        n_sources: Optional[int] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["MNMFbase"], None], List[Callable[["MNMFbase"], None]]]
        ] = None,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            n_basis,
            n_sources=n_sources,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            record_loss=record_loss,
            reference_id=reference_id,
            rng=rng,
        )

    def separate(self, input):
        """Separate ``input`` using multichannel Wiener filter.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_sources, n_bins, n_frames).
        """
        X = input
        T, V = self.basis, self.activation
        H, Z = self.spatial, self.latent

        reference_id = self.reference_id

        ZTV = self.reconstruct_nmf(T, V, Z)
        R_hat_n = ZTV[:, :, :, np.newaxis, np.newaxis] * H[:, :, np.newaxis, :, :]
        R_hat = np.sum(R_hat_n, axis=0)
        R_hat = to_psd(R_hat, flooring_fn=self.flooring_fn)
        R_hat = np.tile(R_hat, reps=(self.n_sources, 1, 1, 1, 1))
        G_Hermite = np.linalg.solve(R_hat, R_hat_n)
        G = G_Hermite.transpose(0, 1, 2, 4, 3).conj()
        G_ref = G[:, :, :, reference_id, :]
        G_ref = G_ref.transpose(0, 3, 1, 2)
        Y = np.sum(G_ref * X, axis=1)

        return Y

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        Returns:
            Computed loss.
        """
        X = self.input
        T, V = self.basis, self.activation
        H, Z = self.spatial, self.latent

        R = X[:, np.newaxis] * X[np.newaxis, :].conj()
        R = R.transpose(2, 3, 0, 1)
        R = to_psd(R, flooring_fn=self.flooring_fn)
        R_hat = self.reconstruct_mnmf(T, V, H, Z)
        R_hat = to_psd(R_hat, flooring_fn=self.flooring_fn)
        RR_inv = np.linalg.solve(R_hat, R)  # Hermitian transpose of R @ np.linalg.inv(R_hat)
        trace = np.trace(RR_inv, axis1=-2, axis2=-1)
        trace = np.real(trace)
        logdet = self.compute_logdet(R, R_hat)
        loss = np.mean(trace - logdet - self.n_channels, axis=-1)
        loss = loss.sum(axis=0)
        loss = loss.item()

        return loss

    def compute_logdet(self, target: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
        r"""Compute log-determinant.

        Args:
            target:
                Covariance matrix with shape of (\*, n_channels, n_channels).
            reconstructed:
                Reconstructed MNMF with shape of (\*, n_channels, n_channels).

        Returns:
            numpy.ndarray of computed log-determinant values.
            The shape is (\*).
        """
        _, logdet_R = np.linalg.slogdet(target)
        _, logdet_R_hat = np.linalg.slogdet(reconstructed)
        logdet = logdet_R - logdet_R_hat

        return logdet
