from typing import Callable, List, Optional, Union

import numpy as np

from ..algorithm import (
    MINIMAL_DISTORTION_PRINCIPLE_KEYWORDS,
    PROJECTION_BACK_KEYWORDS,
    minimal_distortion_principle,
    projection_back,
)
from ..linalg import prox
from .base import IterativeMethodBase

EPS = 1e-10


class PDSBSSBase(IterativeMethodBase):
    r"""Base class of blind source separation \
    via proximal splitting algorithm [#yatabe2018determined]_.

    Args:
        penalty_fn (callable):
            Penalty function that determines source model.
        prox_penalty (callable):
            Proximal operator of penalty function.
            Default: ``None``.
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

    .. [#yatabe2018determined] K. Yatabe and D. Kitamura,
        "Determined blind source separation via proximal splitting algorithm,"
        in *Proc of ICASSP*, pp. 776-780, 2018.
    """

    def __init__(
        self,
        penalty_fn: Callable[[np.ndarray, np.ndarray], float] = None,
        prox_penalty: Callable[[np.ndarray, float], np.ndarray] = None,
        callbacks: Optional[
            Union[Callable[["PDSBSSBase"], None], List[Callable[["PDSBSSBase"], None]]]
        ] = None,
        scale_restoration: bool = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            callbacks=callbacks,
            record_loss=record_loss,
        )

        if penalty_fn is None:
            raise ValueError("Specify penalty function.")
        else:
            if callable(penalty_fn):
                penalty_fn = [penalty_fn]
            self.penalty_fn = penalty_fn

        if prox_penalty is None:
            raise ValueError("Specify proximal operator of penalty function.")
        else:
            if callable(prox_penalty):
                prox_penalty = [prox_penalty]
            self.prox_penalty = prox_penalty

        assert len(self.penalty_fn) == len(
            self.prox_penalty
        ), "Length of penalty_fn and prox_penalty are different."

        self.input = None
        self.scale_restoration = scale_restoration

        if reference_id is None and scale_restoration:
            raise ValueError("Specify 'reference_id' if scale_restoration=True.")
        else:
            self.reference_id = reference_id

    def __repr__(self) -> str:
        s = "PDSBSS("
        s += "n_penalties={n_penalties}".format(n_penalties=self.n_penalties)
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
                Keyword arguments to set as attributes of PDSBSSBase.
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

    @property
    def n_penalties(self):
        r"""Return number of penalty terms."""
        return len(self.penalty_fn)

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

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        Returns:
            Computed loss.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)  # (n_sources, n_bins, n_frames)
        logdet = self.compute_logdet(W)  # (n_bins,)
        penalty = 0

        for penalty_fn in self.penalty_fn:
            penalty = penalty + penalty_fn(Y)

        loss = penalty - np.sum(logdet, axis=0)
        loss = loss.item()

        return loss

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

    def normalize_by_spectral_norm(self, input: np.ndarray, n_penalties: int = None) -> np.ndarray:
        r"""Spectral normalization.

        Args:
            input (numpy.ndarray):
                Input spectrogram with shape of (n_channels, n_bins, n_frames).
            n_penalties (int):
                Number of penalty functions, which determines coefficient of normalization.

        Returns:
            numpy.ndarray of normalized spectrogram with shape of (n_channels, n_bins, n_frames).
        """
        if n_penalties is None:
            n_penalties = self.n_penalties

        norm = np.linalg.norm(input.transpose(1, 0, 2), ord=2, axis=(-2, -1))
        norm = np.max(norm)

        return input / (np.sqrt(n_penalties) * norm)

    def restore_scale(self) -> None:
        r"""Restore scale ambiguity.

        If ``self.scale_restoration=projection_back``, we use projection back technique.
        """
        scale_restoration = self.scale_restoration

        assert scale_restoration, "Set self.scale_restoration=True."

        if type(scale_restoration) is bool:
            scale_restoration = "projection_back"

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


class PDSBSS(PDSBSSBase):
    r"""Blind source separation via proximal splitting algorithm [#yatabe2018determined]_.

    Args:
        mu1 (float):
            Step size. Default: ``1``.
        mu2 (float):
            Step size. Default: ``1``.
        alpha (float):
            Step size. Default: ``1``.
        penalty_fn (callable):
            Penalty function that determines source model.
        prox_penalty (callable):
            Proximal operator of penalty function.
            Default: ``None``.
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
    """

    def __init__(
        self,
        mu1: float = 1,
        mu2: float = 1,
        alpha: float = 1,
        penalty_fn: Callable[[np.ndarray, np.ndarray], float] = None,
        prox_penalty: Callable[[np.ndarray, float], np.ndarray] = None,
        callbacks: Optional[
            Union[Callable[["PDSBSS"], None], List[Callable[["PDSBSS"], None]]]
        ] = None,
        scale_restoration: bool = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            penalty_fn=penalty_fn,
            prox_penalty=prox_penalty,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

        self.mu1, self.mu2 = mu1, mu2
        self.alpha = alpha

    def __call__(self, input, n_iter=100, initial_call: bool = True, **kwargs) -> np.ndarray:
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

        # Call __call__ of PDSBSSBase's parent, i.e. __call__ of IterativeMethodBase
        super(PDSBSSBase, self).__call__(n_iter=n_iter, initial_call=initial_call)

        if self.scale_restoration:
            self.restore_scale()

        self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "PDSBSS("
        s += "mu1={mu1}, mu2={mu2}"
        s += ", alpha={alpha}"
        s += ", n_penalties={n_penalties}".format(n_penalties=self.n_penalties)
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
                Keyword arguments to set as attributes of PDSBSS.
        """
        super()._reset(**kwargs)

        n_penalties = self.n_penalties
        n_sources = self.n_sources
        n_bins, n_frames = self.n_bins, self.n_frames

        if not hasattr(self, "dual"):
            dual = np.zeros((n_penalties, n_sources, n_bins, n_frames), dtype=np.complex128)
        else:
            if self.dual is None:
                dual = None
            else:
                # To avoid overwriting ``dual`` given by keyword arguments.
                dual = self.dual.copy()

        self.dual = dual

    def update_once(self) -> None:
        r"""Update demixing filters and dual parameters once."""
        mu1, mu2 = self.mu1, self.mu2
        alpha = self.alpha

        Y = self.dual
        X, W = self.input, self.demix_filter

        Y_sum = Y.sum(axis=0)
        XY = Y_sum.transpose(1, 0, 2) @ X.transpose(1, 2, 0).conj()
        W_tilde = prox.neg_logdet(W - mu1 * mu2 * XY, step_size=mu1)
        XW = self.separate(X, demix_filter=2 * W_tilde - W)
        Y_tilde = []

        for Y_q, prox_penalty in zip(Y, self.prox_penalty):
            Z_q = Y_q + XW
            Y_tilde_q = Z_q - prox_penalty(Z_q, step_size=1 / mu2)
            Y_tilde.append(Y_tilde_q)

        Y_tilde = np.stack(Y_tilde, axis=0)

        self.demix_filter = alpha * W_tilde + (1 - alpha) * W
        self.dual = alpha * Y_tilde + (1 - alpha) * Y
