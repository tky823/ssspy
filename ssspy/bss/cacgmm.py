from typing import Callable, List, Optional, Union

import numpy as np

from ..linalg.quadratic import quadratic
from ..special.logsumexp import logsumexp
from ..special.softmax import softmax
from .base import IterativeMethodBase


class CACGMMbase(IterativeMethodBase):
    r"""Base class of complex angular central Gaussian mixture model (cACGMM).

    Args:
        n_sources (int, optional):
            Number of sources to be separated.
            If ``None`` is given, ``n_sources`` is determined by number of channels
            in input spectrogram. Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        rng (numpy.random.Generator, optioinal):
            Random number generator. This is mainly used to randomly initialize parameters
            of cACGMM. If ``None`` is given, ``np.random.default_rng()`` is used.
            Default: ``None``.
    """

    def __init__(
        self,
        n_sources: Optional[int] = None,
        callbacks: Optional[
            Union[
                Callable[["CACGMMbase"], None],
                List[Callable[["CACGMMbase"], None]],
            ]
        ] = None,
        record_loss: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(callbacks=callbacks, record_loss=record_loss)

        self.n_sources = n_sources
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

        raise NotImplementedError("Implement '__call__' method.")

    def _reset(self, **kwargs) -> None:
        r"""Reset attributes by given keyword arguments.

        Args:
            kwargs:
                Keyword arguments to set as attributes of CACGMM.
        """
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        X = self.input

        norm = np.linalg.norm(X, axis=0)
        Z = X / norm
        self.unit_input = Z

        n_sources = self.n_sources
        n_channels, n_bins, n_frames = X.shape

        if n_sources is None:
            n_sources = n_channels

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        self._init_parameters(rng=self.rng)

    def _init_parameters(self, rng: Optional[np.random.Generator] = None) -> None:
        r"""Initialize parameters of cACGMM.

        Args:
            rng (numpy.random.Generator, optional):
                Random number generator. If ``None`` is given,
                ``np.random.default_rng()`` is used.
                Default: ``None``.
        """
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins, n_frames = self.n_bins, self.n_frames

        if rng is None:
            rng = np.random.default_rng()

        alpha = np.ones((n_sources, n_bins)) / n_sources
        eye = np.eye(n_channels, dtype=np.complex128)
        B = np.tile(eye, reps=(n_sources, n_bins, 1, 1))
        gamma = rng.random((n_sources, n_bins, n_frames))

        self.mixing = alpha
        self.covariance = B
        self.posterior = gamma / gamma.sum(axis=0)

    def separate(self, input: np.ndarray) -> np.ndarray:
        r"""Separate ``input``.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_sources, n_bins, n_frames).
        """
        raise NotImplementedError("Implement 'separate' method.")

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        Returns:
            Computed loss.
        """
        raise NotImplementedError("Implement 'compute_loss' method.")

    def compute_logdet(self, covariance: np.ndarray) -> np.ndarray:
        r"""Compute log-determinant of input.

        Args:
            covariance (numpy.ndarray):
                Covariance matrix with shape of (n_sources, n_bins, n_channels, n_channels).

        Returns:
            numpy.ndarray of log-determinant.
        """
        _, logdet = np.linalg.slogdet(covariance)

        return logdet


class CACGMM(CACGMMbase):
    r"""Complex angular central Gaussian mixture model (cACGMM) [#ito2016complex]_.

    Args:
        n_sources (int, optional):
            Number of sources to be separated.
            If ``None`` is given, ``n_sources`` is determined by number of channels
            in input spectrogram. Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel to extract separated signals. Default: ``0``.
        rng (numpy.random.Generator, optioinal):
            Random number generator. This is mainly used to randomly initialize parameters
            of cACGMM. If ``None`` is given, ``np.random.default_rng()`` is used.
            Default: ``None``.

    .. [#ito2016complex] N. Ito, S. Araki, and T. Nakatani. \
        "Complex angular central Gaussian mixture model for directional statistics \
        in mask-based microphone array signal processing,"
        in *Proc. EUSIPCO*, 2016, pp. 1153-1157.
    """

    def __init__(
        self,
        n_sources: Optional[int] = None,
        callbacks: Optional[
            Union[
                Callable[["CACGMM"], None],
                List[Callable[["CACGMM"], None]],
            ]
        ] = None,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(n_sources=n_sources, callbacks=callbacks, record_loss=record_loss, rng=rng)

        self.reference_id = reference_id

    def update_parameters(self) -> None:
        r"""Update parameters of mixture of complex angular central Gaussian distributions.

        This method corresponds to M step in EM algorithm for cACGMM.
        """
        alpha = self.mixing
        Z = self.unit_input
        B = self.covariance
        gamma = self.posterior

        Z = Z.transpose(1, 2, 0)
        B_inverse = np.linalg.inv(B)
        ZBZ = quadratic(Z, B_inverse[:, :, np.newaxis])
        ZBZ = np.real(ZBZ)
        ZZ = Z[:, :, :, np.newaxis] * Z[:, :, np.newaxis, :].conj()

        alpha = np.mean(gamma, axis=-1)

        GZBZ = gamma / ZBZ
        num = np.sum(GZBZ[:, :, :, np.newaxis, np.newaxis] * ZZ, axis=2)
        denom = np.sum(gamma, axis=2)
        B = self.n_channels * (num / denom[:, :, np.newaxis, np.newaxis])

        self.mixing = alpha
        self.covariance = B

    def update_posterior(self) -> None:
        r"""Update posteriors.

        This method corresponds to E step in EM algorithm for cACGMM.
        """
        alpha = self.mixing
        Z = self.unit_input
        B = self.covariance

        Z = Z.transpose(1, 2, 0)
        B_inverse = np.linalg.inv(B)
        ZBZ = quadratic(Z, B_inverse[:, :, np.newaxis])
        ZBZ = np.real(ZBZ)

        log_prob = np.log(alpha) - self.compute_logdet(B)
        log_gamma = log_prob[:, :, np.newaxis] - self.n_channels * np.log(ZBZ)

        gamma = softmax(log_gamma, axis=0)

        self.posterior = gamma

    def compute_loss(self) -> float:
        r"""Compute loss of cACGMM :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is defined as follows:

        .. math::
            \mathcal{L}
            = -\frac{1}{J}\sum_{i,j}\log\left(
            \sum_{n}\frac{\alpha_{in}}{\det\boldsymbol{B}_{in}}
            \frac{1}{(\boldsymbol{z}_{ij}^{\mathsf{H}}\boldsymbol{B}_{in}\boldsymbol{z}_{ij})^{M}}
            \right).
        """
        alpha = self.mixing
        Z = self.unit_input
        B = self.covariance

        Z = Z.transpose(1, 2, 0)
        B_inverse = np.linalg.inv(B)
        ZBZ = quadratic(Z, B_inverse[:, :, np.newaxis])
        ZBZ = np.real(ZBZ)

        log_prob = np.log(alpha) - self.compute_logdet(B)
        log_gamma = log_prob[:, :, np.newaxis] - self.n_channels * np.log(ZBZ)

        loss = -logsumexp(log_gamma, axis=0)
        loss = np.mean(loss, axis=-1)
        loss = loss.sum(axis=0)
        loss = loss.item()

        return loss
