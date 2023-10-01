import warnings
from typing import Callable, List, Optional, Union

import numpy as np

from ..linalg import prox
from .proxbss import ProxBSSBase

EPS = 1e-10


class ADMMBSSBase(ProxBSSBase):
    """Base class of blind source separation via alternative direction method of multiplier.

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

    """

    def __repr__(self) -> str:
        s = "ADMMBSS("
        s += "n_penalties={n_penalties}".format(n_penalties=self.n_penalties)
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)


class ADMMBSS(ADMMBSSBase):
    """Base class of blind source separation via alternative direction method of multiplier.

    Args:
        rho (float):
            Penalty parameter. Default: ``1``.
        alpha (float):
            Relaxation parameter (deprecated). Set ``relaxation`` instead.
        relaxation (float):
            Relaxation parameter. Default: ``1``.
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
        rho: float = 1,
        alpha: float = None,
        relaxation: float = 1,
        penalty_fn: Callable[[np.ndarray, np.ndarray], float] = None,
        prox_penalty: Callable[[np.ndarray, float], np.ndarray] = None,
        callbacks: Optional[
            Union[Callable[["ADMMBSS"], None], List[Callable[["ADMMBSS"], None]]]
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

        self.rho = rho

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

        # Call __call__ of ADMMBSSBase's parent, i.e. __call__ of IterativeMethodBase
        super(ADMMBSSBase, self).__call__(n_iter=n_iter, initial_call=initial_call)

        if self.scale_restoration:
            self.restore_scale()

        self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "ADMMBSS("
        s += "rho={rho}"
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
                Keyword arguments to set as attributes of ADMMBSS.
        """
        if "aux1" in kwargs.keys():
            warnings.warn("aux1 is deprecated. Use auxiliary1 instead.", DeprecationWarning)

            kwargs["auxiliary1"] = kwargs.pop("aux1")

        if "aux2" in kwargs.keys():
            warnings.warn("aux2 is deprecated. Use auxiliary2 instead.", DeprecationWarning)

            kwargs["auxiliary2"] = kwargs.pop("aux2")

        super()._reset(**kwargs)

        n_penalties = self.n_penalties
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins, n_frames = self.n_bins, self.n_frames

        if not hasattr(self, "auxiliary1"):
            auxiliary1 = np.zeros((n_bins, n_sources, n_channels), dtype=np.complex128)
        else:
            # To avoid overwriting ``auxiliary1`` given by keyword arguments.
            auxiliary1 = self.auxiliary1.copy()

        if not hasattr(self, "auxiliary2"):
            auxiliary2 = np.zeros((n_penalties, n_sources, n_bins, n_frames), dtype=np.complex128)
        else:
            # To avoid overwriting ``auxiliary2`` given by keyword arguments.
            auxiliary2 = self.auxiliary2.copy()

        if not hasattr(self, "dual1"):
            dual1 = np.zeros((n_bins, n_sources, n_channels), dtype=np.complex128)
        else:
            # To avoid overwriting ``dual1`` given by keyword arguments.
            dual1 = self.dual1.copy()

        if not hasattr(self, "dual2"):
            dual2 = np.zeros((n_penalties, n_sources, n_bins, n_frames), dtype=np.complex128)
        else:
            # To avoid overwriting ``dual2`` given by keyword arguments.
            dual2 = self.dual2.copy()

        self.auxiliary1 = auxiliary1
        self.auxiliary2 = auxiliary2
        self.dual1 = dual1
        self.dual2 = dual2

    def update_once(self) -> None:
        r"""Update demixing filters, auxiliary parameters, and dual parameters once."""
        n_penalties = self.n_penalties
        n_channels = self.n_channels
        rho, alpha = self.rho, self.relaxation

        V, V_tilde = self.auxiliary1, self.auxiliary2
        Y, Y_tilde = self.dual1, self.dual2
        X, W = self.input, self.demix_filter

        XX = X.transpose(1, 0, 2).conj() @ X.transpose(1, 2, 0)
        E = np.eye(n_channels)
        VY = V - Y
        VY_tilde = np.sum(V_tilde - Y_tilde, axis=0)
        XVY_tilde = X.transpose(1, 0, 2).conj() @ VY_tilde.transpose(1, 2, 0)

        W = np.linalg.solve(n_penalties * XX + E, VY + XVY_tilde.transpose(0, 2, 1))
        XW = self.separate(X, demix_filter=W)

        U = alpha * W + (1 - alpha) * V
        U_tilde = alpha * XW + (1 - alpha) * V_tilde

        V = prox.neg_logdet(U + Y, step_size=1 / rho)

        V_tilde = []

        for U_tilde_q, Y_tilde_q, prox_penalty in zip(U_tilde, Y_tilde, self.prox_penalty):
            V_tilde_q = prox_penalty(U_tilde_q + Y_tilde_q, step_size=1 / rho)
            V_tilde.append(V_tilde_q)

        V_tilde = np.stack(V_tilde, axis=0)

        Y = Y + U - V
        Y_tilde = Y_tilde + U_tilde - V_tilde

        self.auxiliary1, self.auxiliary2 = V, V_tilde
        self.dual1, self.dual2 = Y, Y_tilde
        self.demix_filter = W
