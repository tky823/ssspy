import math
from typing import Callable, List, Optional, Union

import numpy as np

from .pdsbss import MaskingPDSBSS

__all__ = [
    "MaskingPDSHVA",
    "HVA",
]

EPS = 1e-10


class MaskingPDSHVA(MaskingPDSBSS):
    r"""Harmonic vector analysis.

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
        in *Proc of ICASSP*, pp. 715-719, 2019.

    """

    def __init__(
        self,
        mu1: float = 1,
        mu2: float = 1,
        alpha: float = None,
        relaxation: float = 1,
        attenuation: Optional[float] = None,
        mask_iter: int = 1,
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
            gamma = attenuation

            if gamma is None:
                gamma = 1 / n_sources

            # TODO: custom flooring function
            zeta = np.log(np.abs(y) + EPS)
            zeta_mean = zeta.mean(axis=1, keepdims=True)
            rho = zeta - zeta_mean
            nu = np.fft.irfft(
                rho, axis=1, norm="backward"
            )  # (n_sources, 2 * (n_bins - 1), n_frames).
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


class HVA(MaskingPDSHVA):
    """Alias of MaskingPDSHVA."""
