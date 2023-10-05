import functools
import math
from typing import Callable, List, Optional, Union

import numpy as np

from ..special.flooring import identity, max_flooring
from .pdsbss import MaskingPDSBSS

__all__ = [
    "MaskingPDSHVA",
    "HVA",
]

EPS = 1e-10


class MaskingPDSHVA(MaskingPDSBSS):
    r"""Harmonic vector analysis proposed in [#yatabe2021determined]_.

    Args:
        mu1 (float):
            Step size. Default: ``1``.
        mu2 (float):
            Step size. Default: ``1``.
        alpha (float):
            Relaxation parameter (deprecated). Set ``relaxation`` instead.
        relaxation (float):
            Relaxation parameter. Default: ``1``.
        attenuation (float, optional):
            Attenuation parameter in masking. Default: ``1 / n_sources``.
        mask_iter (int):
            Number of iterations in application of cosine shrinkage operator.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
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

    .. [#yatabe2021determined] K. Yatabe and D. Kitamura,
        "Determined BSS based on time-frequency masking and its application to \
        harmonic vector analysis," *IEEE/ACM Trans. ASLP*, vol. 29, pp. 1609-1625, 2021.

    """

    def __init__(
        self,
        mu1: float = 1,
        mu2: float = 1,
        alpha: float = None,
        relaxation: float = 1,
        attenuation: Optional[float] = None,
        mask_iter: int = 1,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["MaskingPDSHVA"], None], List[Callable[["MaskingPDSHVA"], None]]]
        ] = None,
        scale_restoration: bool = True,
        record_loss: Optional[bool] = None,
        reference_id: int = 0,
    ) -> None:
        def mask_fn(y: np.ndarray) -> np.ndarray:
            """Masking function to emphasize harmonic components.

            Args:
                y (np.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                np.ndarray of mask. The shape is (n_sources, n_bins, n_frames).
            """
            n_sources, n_bins, _ = y.shape

            if self.attenuation is None:
                self.attenuation = 1 / n_sources

            gamma = self.attenuation

            y = self.flooring_fn(np.abs(y))
            zeta = np.log(y)
            zeta_mean = zeta.mean(axis=1, keepdims=True)
            rho = zeta - zeta_mean
            nu = np.fft.irfft(rho, axis=1, norm="backward")
            nu = nu[:, :n_bins]
            varsigma = np.minimum(1, nu)

            for _ in range(mask_iter):
                varsigma = (1 - np.cos(math.pi * varsigma)) / 2

            xi = np.fft.irfft(varsigma * nu, axis=1, norm="forward")
            xi = xi[:, :n_bins]
            varrho = xi + zeta_mean
            v = np.exp(2 * varrho)
            mask = (v / v.sum(axis=0)) ** gamma

            return mask

        super().__init__(
            mu1=mu1,
            mu2=mu2,
            alpha=alpha,
            relaxation=relaxation,
            penalty_fn=None,
            mask_fn=mask_fn,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

        self.attenuation = attenuation
        self.mask_iter = mask_iter

        if flooring_fn is None:
            self.flooring_fn = identity
        else:
            self.flooring_fn = flooring_fn

    def __repr__(self) -> str:
        s = "MaskingPDSHVA("
        s += "mu1={mu1}, mu2={mu2}"
        s += ", relaxation={relaxation}"

        if self.attenuation is not None:
            s += ", attenuation={attenuation}"

        s += ", mask_iter={mask_iter}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)


class HVA(MaskingPDSHVA):
    """Alias of MaskingPDSHVA."""

    def __repr__(self) -> str:
        s = "HVA("
        s += "mu1={mu1}, mu2={mu2}"
        s += ", relaxation={relaxation}"

        if self.attenuation is not None:
            s += ", attenuation={attenuation}"

        s += ", mask_iter={mask_iter}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)
