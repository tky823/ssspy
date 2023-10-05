import warnings
from typing import Callable, List, Optional, Union

import numpy as np

from ..linalg import prox
from .proxbss import ProxBSSBase

EPS = 1e-10

__all__ = ["PDSBSS", "MaskingPDSBSS"]


class PDSBSSBase(ProxBSSBase):
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
        in *Proc. ICASSP*, 2018, pp. 776-780.
    """

    def __repr__(self) -> str:
        s = "PDSBSS("
        s += "n_penalties={n_penalties}".format(n_penalties=self.n_penalties)
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)


class PDSBSS(PDSBSSBase):
    r"""Blind source separation via proximal splitting algorithm [#yatabe2018determined]_.

    Args:
        mu1 (float):
            Step size. Default: ``1``.
        mu2 (float):
            Step size. Default: ``1``.
        alpha (float):
            Relaxation parameter (deprecated). Set ``relaxation`` instead.
        relaxation (float):
            Relaxation parameter. Default: ``1``.
        penalty_fn (callable, optional):
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
        record_loss (bool, optional):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``None``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.
    """

    def __init__(
        self,
        mu1: float = 1,
        mu2: float = 1,
        alpha: float = None,
        relaxation: float = 1,
        penalty_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        prox_penalty: Callable[[np.ndarray, float], np.ndarray] = None,
        callbacks: Optional[
            Union[Callable[["PDSBSS"], None], List[Callable[["PDSBSS"], None]]]
        ] = None,
        scale_restoration: bool = True,
        record_loss: Optional[bool] = None,
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

        if alpha is None:
            self.relaxation = relaxation
        else:
            assert relaxation == 1, "You cannot specify relaxation and alpha simultaneously."

            warnings.warn("alpha is deprecated. Set relaxation instead.", DeprecationWarning)

            self.relaxation = alpha

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
        s += ", relaxation={relaxation}"
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
        alpha = self.relaxation

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


class MaskingPDSBSS(PDSBSSBase):
    r"""Blind source separation via proximal splitting algorithm with masking [#yatabe2019time]_.

    Args:
        mu1 (float):
            Step size. Default: ``1``.
        mu2 (float):
            Step size. Default: ``1``.
        alpha (float):
            Relaxation parameter (deprecated). Set ``relaxation`` instead.
        relaxation (float):
            Relaxation parameter. Default: ``1``.
        penalty_fn (callable, optional):
            Penalty function that determines source model.
        mask_fn (callable):
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

    .. [#yatabe2019time] K. Yatabe and D. Kitamura,
        "Time-frequency-masking-based determined BSS with application to sparse IVA,"
        in *Proc. ICASSP*, pp. 715-719, 2019.

    """

    def __init__(
        self,
        mu1: float = 1,
        mu2: float = 1,
        alpha: float = None,
        relaxation: float = 1,
        penalty_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        mask_fn: Callable[[np.ndarray], float] = None,
        callbacks: Optional[
            Union[Callable[["MaskingPDSBSS"], None], List[Callable[["MaskingPDSBSS"], None]]]
        ] = None,
        scale_restoration: bool = True,
        record_loss: Optional[bool] = None,
        reference_id: int = 0,
    ) -> None:
        super(ProxBSSBase, self).__init__(
            callbacks=callbacks,
            record_loss=record_loss,
        )

        if penalty_fn is None:
            # Since penalty_fn is not necessarily written in closed form,
            # None is acceptable.
            if record_loss is None:
                record_loss = False

            assert not record_loss, "To record loss, set penalty_fn."
        else:
            assert callable(penalty_fn), "penalty_fn should be callable."

            if record_loss is None:
                record_loss = True

        if mask_fn is None:
            raise ValueError("Specify masking function.")
        else:
            assert callable(mask_fn), "mask_fn should be callable."

        self.penalty_fn = penalty_fn
        self.mask_fn = mask_fn

        self.input = None
        self.scale_restoration = scale_restoration

        if reference_id is None and scale_restoration:
            raise ValueError("Specify 'reference_id' if scale_restoration=True.")
        else:
            self.reference_id = reference_id

        self.mu1, self.mu2 = mu1, mu2

        if alpha is None:
            self.relaxation = relaxation
        else:
            assert relaxation == 1, "You cannot specify relaxation and alpha simultaneously."

            warnings.warn("alpha is deprecated. Set relaxation instead.", DeprecationWarning)

            self.relaxation = alpha

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
        s = "MaskingPDSBSS("
        s += "mu1={mu1}, mu2={mu2}"
        s += ", relaxation={relaxation}"
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
                Keyword arguments to set as attributes of MaskingPDSBSS.
        """
        super()._reset(**kwargs)

        assert self.n_penalties == 1, "Number of penalty function should be one."

        n_sources = self.n_sources
        n_bins, n_frames = self.n_bins, self.n_frames

        if not hasattr(self, "dual"):
            dual = np.zeros((n_sources, n_bins, n_frames), dtype=np.complex128)
        else:
            if self.dual is None:
                dual = None
            else:
                # To avoid overwriting ``dual`` given by keyword arguments.
                dual = self.dual.copy()

        self.dual = dual

    @property
    def n_penalties(self):
        r"""Return number of penalty terms."""
        return 1

    def update_once(self) -> None:
        r"""Update demixing filters and dual parameters once."""
        mu1, mu2 = self.mu1, self.mu2
        alpha = self.relaxation

        Y = self.dual
        X, W = self.input, self.demix_filter

        XY = Y.transpose(1, 0, 2) @ X.transpose(1, 2, 0).conj()
        W_tilde = prox.neg_logdet(W - mu1 * mu2 * XY, step_size=mu1)
        XW = self.separate(X, demix_filter=2 * W_tilde - W)

        Z = Y + XW
        Y_tilde = Z - self.mask_fn(Z) * Z

        self.demix_filter = alpha * W_tilde + (1 - alpha) * W
        self.dual = alpha * Y_tilde + (1 - alpha) * Y
