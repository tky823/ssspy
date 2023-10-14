import functools
from typing import Callable, List, Optional, Union

import numpy as np

from ..algorithm.permutation_alignment import (
    correlation_based_permutation_solver,
    score_based_permutation_solver,
)
from ..linalg.quadratic import quadratic
from ..special.flooring import identity, max_flooring
from ..special.logsumexp import logsumexp
from ..special.psd import to_psd
from ..special.softmax import softmax
from ..utils.flooring import choose_flooring_fn
from .base import IterativeMethodBase

EPS = 1e-10


class CACGMMBase(IterativeMethodBase):
    r"""Base class of complex angular central Gaussian mixture model (cACGMM).

    Args:
        n_sources (int, optional):
            Number of sources to be separated.
            If ``None`` is given, ``n_sources`` is determined by number of channels
            in input spectrogram. Default: ``None``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
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
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[
                Callable[["CACGMMBase"], None],
                List[Callable[["CACGMMBase"], None]],
            ]
        ] = None,
        record_loss: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.normalization: bool
        self.permutation_alignment: bool

        super().__init__(callbacks=callbacks, record_loss=record_loss)

        self.n_sources = n_sources

        if flooring_fn is None:
            self.flooring_fn = identity
        else:
            self.flooring_fn = flooring_fn

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

        raise NotImplementedError("Implement '__call__' method.")

    def __repr__(self) -> str:
        s = "CACGMM("

        if self.n_sources is not None:
            s += "n_sources={n_sources}, "

        s += "record_loss={record_loss}"

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
                Keyword arguments to set as attributes of CACGMM.
        """
        assert self.input is not None, "Specify data!"

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        X = self.input

        norm = np.linalg.norm(X, axis=0)
        Z = X / flooring_fn(norm)
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

        .. note::

            Custom initialization is not supported now.

        """
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins

        if rng is None:
            rng = np.random.default_rng()

        alpha = rng.random((n_sources, n_bins))
        alpha = alpha / alpha.sum(axis=0)

        eye = np.eye(n_channels, dtype=np.complex128)
        B_diag = self.rng.random((n_sources, n_bins, n_channels))
        B_diag = B_diag / B_diag.sum(axis=-1, keepdims=True)
        B = B_diag[:, :, :, np.newaxis] * eye

        self.mixing = alpha
        self.covariance = B

        # The shape of posterior is (n_sources, n_bins, n_frames).
        # This is always required to satisfy posterior.sum(axis=0) = 1
        self.posterior = None

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

    def normalize_covariance(self) -> None:
        r"""Normalize covariance of cACG.

        .. math::
            \boldsymbol{B}_{in}
            \leftarrow\frac{\boldsymbol{B}_{in}}{\mathrm{tr}(\boldsymbol{B}_{in})}
        """
        assert self.normalization, "Set normalization."

        B = self.covariance

        trace = np.trace(B, axis1=-2, axis2=-1)
        trace = np.real(trace)
        B = B / trace[..., np.newaxis, np.newaxis]

        self.covariance = B

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

    def solve_permutation(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Align posteriors and separated spectrograms.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """

        permutation_alignment = self.permutation_alignment
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        assert permutation_alignment, "Set permutation_alignment=True."

        if type(permutation_alignment) is bool:
            # when permutation_alignment is True
            permutation_alignment = "posterior_score"

        if permutation_alignment in ["posterior_score", "posterior_correlation"]:
            target = "posterior"
        elif permutation_alignment in ["amplitude_score", "amplitude_correlation"]:
            target = "amplitude"
        else:
            raise NotImplementedError(
                "permutation_alignment {} is not implemented.".format(permutation_alignment)
            )

        if permutation_alignment in ["posterior_score", "amplitude_score"]:
            self.solve_permutation_by_score(target=target, flooring_fn=flooring_fn)
        elif permutation_alignment in ["posterior_correlation", "amplitude_correlation"]:
            self.solve_permutation_by_correlation(target=target, flooring_fn=flooring_fn)
        else:
            raise NotImplementedError(
                "permutation_alignment {} is not implemented.".format(permutation_alignment)
            )

    def solve_permutation_by_score(
        self,
        target: str = "posterior",
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Align posteriors and amplitudes of separated spectrograms by score value.

        Args:
            target (str):
                Target to compute score values. Choose ``posterior`` or ``amplitude``.
                Default: ``posterior``.
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.
        """

        assert target in ["posterior", "amplitude"], "Invalid target {} is specified.".format(
            target
        )

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        X = self.input
        alpha = self.mixing
        B = self.covariance
        gamma = self.posterior

        if hasattr(self, "global_iter"):
            global_iter = self.global_iter
        else:
            global_iter = 1

        if hasattr(self, "local_iter"):
            local_iter = self.local_iter
        else:
            local_iter = 1

        Y = self.separate(X, posterior=gamma)

        alpha = alpha.transpose(1, 0)
        B = B.transpose(1, 0, 2, 3)
        gamma = gamma.transpose(1, 0, 2)

        if target == "posterior":
            gamma, (alpha, B) = score_based_permutation_solver(
                gamma,
                alpha,
                B,
                global_iter=global_iter,
                local_iter=local_iter,
                flooring_fn=flooring_fn,
            )
        elif target == "amplitude":
            Y = Y.transpose(1, 0, 2)
            amplitude = np.abs(Y)

            _, (alpha, B, gamma) = score_based_permutation_solver(
                amplitude,
                alpha,
                B,
                gamma,
                global_iter=global_iter,
                local_iter=local_iter,
                flooring_fn=flooring_fn,
            )
        else:
            raise ValueError("Invalid target {} is specified.".format(target))

        alpha = alpha.transpose(1, 0)
        B = B.transpose(1, 0, 2, 3)
        gamma = gamma.transpose(1, 0, 2)

        Y = self.separate(X, posterior=gamma)

        self.mixing = alpha
        self.covariance = B
        self.posterior = gamma
        self.output = Y

    def solve_permutation_by_correlation(
        self,
        target: str = "amplitude",
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Align posteriors and amplitudes of separated spectrograms by correlation.

        Args:
            target (str):
                Target to compute correlations. Choose ``posterior`` or ``amplitude``.
                Default: ``amplitude``.
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """

        assert target == "amplitude", "Only amplitude is supported as target."

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        X = self.input
        alpha = self.mixing
        B = self.covariance
        gamma = self.posterior

        Y = self.separate(X, posterior=self.posterior)

        alpha = alpha.transpose(1, 0)
        B = B.transpose(1, 0, 2, 3)
        gamma = gamma.transpose(1, 0, 2)
        Y = Y.transpose(1, 0, 2)
        Y, (alpha, B, gamma) = correlation_based_permutation_solver(
            Y, alpha, B, gamma, flooring_fn=flooring_fn
        )
        alpha = alpha.transpose(1, 0)
        B = B.transpose(1, 0, 2, 3)
        gamma = gamma.transpose(1, 0, 2)
        Y = Y.transpose(1, 0, 2)

        self.mixing = alpha
        self.covariance = B
        self.posterior = gamma
        self.output = Y


class CACGMM(CACGMMBase):
    r"""Complex angular central Gaussian mixture model (cACGMM) [#ito2016complex]_.

    Args:
        n_sources (int, optional):
            Number of sources to be separated.
            If ``None`` is given, ``n_sources`` is determined by number of channels
            in input spectrogram. Default: ``None``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        normalization (bool):
            If ``True`` is given, normalization is applied to covariance in cACG.
        permutation_alignment (bool):
            If ``permutation_alignment=True``, a permutation solver is used to align
            estimated spectrograms. Default: ``True``.
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
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[
                Callable[["CACGMM"], None],
                List[Callable[["CACGMM"], None]],
            ]
        ] = None,
        normalization: bool = True,
        permutation_alignment: bool = True,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_sources=n_sources,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            record_loss=record_loss,
            rng=rng,
        )

        self.normalization = normalization
        self.permutation_alignment = permutation_alignment
        self.reference_id = reference_id

        if type(permutation_alignment) is bool and permutation_alignment:
            valid_keys = {"global_iter", "local_iter"}
        elif type(permutation_alignment) is str and permutation_alignment in [
            "posterior_score",
            "amplitude_score",
        ]:
            valid_keys = {"global_iter", "local_iter"}
        else:
            valid_keys = set()

        invalid_keys = set(kwargs) - valid_keys

        assert invalid_keys == set(), "Invalid keywords {} are given.".format(invalid_keys)

        for key, value in kwargs.items():
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

        # Call __call__ of CACGMMBase's parent, i.e. __call__ of IterativeMethodBase
        super(CACGMMBase, self).__call__(n_iter=n_iter, initial_call=initial_call)

        # posterior should be updated
        self.update_posterior(flooring_fn=self.flooring_fn)

        if self.permutation_alignment:
            self.solve_permutation(flooring_fn=self.flooring_fn)

        X = self.input
        self.output = self.separate(X, posterior=self.posterior)

        return self.output

    def __repr__(self) -> str:
        s = "CACGMM("

        if self.n_sources is not None:
            s += "n_sources={n_sources}, "

        s += "record_loss={record_loss}"
        s += ", normalization={normalization}"
        s += ", permutation_alignment={permutation_alignment}"
        s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def separate(self, input: np.ndarray, posterior: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Separate ``input`` using posterior probabilities.

        In this method, ``self.posterior`` is not updated.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            posterior (numpy.ndarray, optional):
                Posterior probability. If not specified, ``posterior`` is computed by current
                parameters.

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_sources, n_bins, n_frames).
        """
        X = input

        if posterior is None:
            alpha = self.mixing
            Z = self.unit_input
            B = self.covariance

            Z = Z.transpose(1, 2, 0)
            B_inverse = np.linalg.inv(B)
            ZBZ = quadratic(Z, B_inverse[:, :, np.newaxis])
            ZBZ = np.real(ZBZ)
            ZBZ = np.maximum(ZBZ, 0)
            ZBZ = self.flooring_fn(ZBZ)

            log_alpha = np.log(alpha)
            _, logdet = np.linalg.slogdet(B)
            log_prob = log_alpha - logdet
            log_gamma = log_prob[:, :, np.newaxis] - self.n_channels * np.log(ZBZ)

            gamma = softmax(log_gamma, axis=0)
        else:
            gamma = posterior

        return gamma * X[self.reference_id]

    def update_once(
        self, flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self"
    ) -> None:
        r"""Perform E and M step once.

        In ``update_posterior``, posterior probabilities are updated, which corresponds to E step.
        In ``update_parameters``, parameters of cACGMM are updated, which corresponds to M step.

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

        self.update_posterior(flooring_fn=flooring_fn)
        self.update_parameters(flooring_fn=flooring_fn)

        if self.normalization:
            self.normalize_covariance()

    def update_posterior(
        self, flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self"
    ) -> None:
        r"""Update posteriors.

        This method corresponds to E step in EM algorithm for cACGMM.

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

        alpha = self.mixing
        Z = self.unit_input
        B = self.covariance

        Z = Z.transpose(1, 2, 0)
        B_inverse = np.linalg.inv(B)
        ZBZ = quadratic(Z, B_inverse[:, :, np.newaxis])
        ZBZ = np.real(ZBZ)
        ZBZ = np.maximum(ZBZ, 0)
        ZBZ = flooring_fn(ZBZ)

        log_prob = np.log(alpha) - self.compute_logdet(B)
        log_gamma = log_prob[:, :, np.newaxis] - self.n_channels * np.log(ZBZ)

        gamma = softmax(log_gamma, axis=0)

        self.posterior = gamma

    def update_parameters(
        self, flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self"
    ) -> None:
        r"""Update parameters of mixture of complex angular central Gaussian distributions.

        This method corresponds to M step in EM algorithm for cACGMM.

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

        Z = self.unit_input
        B = self.covariance
        gamma = self.posterior

        Z = Z.transpose(1, 2, 0)
        B_inverse = np.linalg.inv(B)
        ZBZ = quadratic(Z, B_inverse[:, :, np.newaxis])
        ZBZ = np.real(ZBZ)
        ZBZ = np.maximum(ZBZ, 0)
        ZBZ = flooring_fn(ZBZ)
        ZZ = Z[:, :, :, np.newaxis] * Z[:, :, np.newaxis, :].conj()

        alpha = np.mean(gamma, axis=-1)

        GZBZ = gamma / ZBZ
        num = np.sum(GZBZ[:, :, :, np.newaxis, np.newaxis] * ZZ, axis=2)
        denom = np.sum(gamma, axis=2)
        B = self.n_channels * (num / denom[:, :, np.newaxis, np.newaxis])
        B = to_psd(B, flooring_fn=flooring_fn)

        self.mixing = alpha
        self.covariance = B

    def compute_loss(self) -> float:
        r"""Compute loss of cACGMM :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is defined as follows:

        .. math::
            \mathcal{L}
            = -\frac{1}{J}\sum_{i,j}\log\left(
            \sum_{n}\frac{\alpha_{in}}{\det\boldsymbol{B}_{in}}
            \frac{1}{(\boldsymbol{z}_{ij}^{\mathsf{H}}\boldsymbol{B}_{in}^{-1}\boldsymbol{z}_{ij})^{M}}
            \right).
        """
        alpha = self.mixing
        Z = self.unit_input
        B = self.covariance

        Z = Z.transpose(1, 2, 0)
        B_inverse = np.linalg.inv(B)
        ZBZ = quadratic(Z, B_inverse[:, :, np.newaxis])
        ZBZ = np.real(ZBZ)
        ZBZ = np.maximum(ZBZ, 0)
        ZBZ = self.flooring_fn(ZBZ)

        log_prob = np.log(alpha) - self.compute_logdet(B)
        log_gamma = log_prob[:, :, np.newaxis] - self.n_channels * np.log(ZBZ)

        loss = -logsumexp(log_gamma, axis=0)
        loss = np.mean(loss, axis=-1)
        loss = loss.sum(axis=0)
        loss = loss.item()

        return loss
