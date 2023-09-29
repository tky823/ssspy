import numpy as np

from ssspy.bss.base import IterativeMethodBase
from ssspy.bss.cacgmm import CACGMM
from ssspy.bss.fdica import (
    AuxFDICA,
    AuxLaplaceFDICA,
    GradFDICA,
    GradLaplaceFDICA,
    NaturalGradFDICA,
    NaturalGradLaplaceFDICA,
)
from ssspy.bss.ica import FastICA, GradICA, GradLaplaceICA, NaturalGradICA, NaturalGradLaplaceICA
from ssspy.bss.ilrma import GGDILRMA, TILRMA, GaussILRMA
from ssspy.bss.ipsdta import TIPSDTA, GaussIPSDTA
from ssspy.bss.iva import (
    PDSIVA,
    AuxGaussIVA,
    AuxIVA,
    AuxLaplaceIVA,
    FasterIVA,
    FastIVA,
    GradGaussIVA,
    GradIVA,
    GradLaplaceIVA,
    NaturalGradGaussIVA,
    NaturalGradIVA,
    NaturalGradLaplaceIVA,
)
from ssspy.bss.mnmf import FastGaussMNMF, GaussMNMF
from ssspy.bss.pdsbss import PDSBSS


def test_grad_ica_inheritance() -> None:
    def contrast_fn(x):
        return np.log(1 + np.exp(x))

    def score_fn(x):
        return 1 / (1 + np.exp(-x))

    ica = GradICA(contrast_fn=contrast_fn, score_fn=score_fn)

    assert isinstance(ica, IterativeMethodBase)

    ica = GradLaplaceICA()

    assert isinstance(ica, IterativeMethodBase)


def test_natural_grad_ica_inheritance() -> None:
    def contrast_fn(x):
        return np.log(1 + np.exp(x))

    def score_fn(x):
        return 1 / (1 + np.exp(-x))

    ica = NaturalGradICA(contrast_fn=contrast_fn, score_fn=score_fn)

    assert isinstance(ica, IterativeMethodBase)

    ica = NaturalGradLaplaceICA()

    assert isinstance(ica, IterativeMethodBase)


def test_fast_ica_inheritance() -> None:
    def contrast_fn(x):
        return np.log(1 + np.exp(x))

    def score_fn(x):
        return 1 / (1 + np.exp(-x))

    def d_score_fn(x):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)

    ica = FastICA(contrast_fn=contrast_fn, score_fn=score_fn, d_score_fn=d_score_fn)

    assert isinstance(ica, IterativeMethodBase)


def test_grad_fdica_inheritance() -> None:
    def contrast_fn(y):
        return 2 * np.abs(y)

    def score_fn(y):
        denominator = np.maximum(np.abs(y), 1e-10)
        return y / denominator

    fdica = GradFDICA(contrast_fn=contrast_fn, score_fn=score_fn)

    assert isinstance(fdica, IterativeMethodBase)

    fdica = GradLaplaceFDICA()

    assert isinstance(fdica, IterativeMethodBase)


def test_natural_grad_fdica_inheritance() -> None:
    def contrast_fn(y):
        return 2 * np.abs(y)

    def score_fn(y):
        denominator = np.maximum(np.abs(y), 1e-10)
        return y / denominator

    fdica = NaturalGradFDICA(contrast_fn=contrast_fn, score_fn=score_fn)

    assert isinstance(fdica, IterativeMethodBase)

    fdica = NaturalGradLaplaceFDICA()

    assert isinstance(fdica, IterativeMethodBase)


def test_aux_fdica_inheritance() -> None:
    def contrast_fn(y):
        return 2 * np.abs(y)

    def d_contrast_fn(y):
        return 2 * np.ones_like(y)

    fdica = AuxFDICA(
        contrast_fn=contrast_fn,
        d_contrast_fn=d_contrast_fn,
    )

    assert isinstance(fdica, IterativeMethodBase)

    fdica = AuxLaplaceFDICA()

    assert isinstance(fdica, IterativeMethodBase)


def test_grad_iva_inheritance() -> None:
    def contrast_fn(y: np.ndarray) -> np.ndarray:
        r"""Contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.linalg.norm(y, axis=1)

    def score_fn(y) -> np.ndarray:
        r"""Score function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_bins, n_frames).
        """
        norm = np.linalg.norm(y, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-10)
        return y / norm

    iva = GradIVA(contrast_fn=contrast_fn, score_fn=score_fn)

    assert isinstance(iva, IterativeMethodBase)

    iva = GradLaplaceIVA()

    assert isinstance(iva, IterativeMethodBase)

    iva = GradGaussIVA()

    assert isinstance(iva, IterativeMethodBase)


def test_natural_grad_iva_inheritance() -> None:
    def contrast_fn(y: np.ndarray) -> np.ndarray:
        r"""Contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.linalg.norm(y, axis=1)

    def score_fn(y):
        r"""Score function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_bins, n_frames).
        """
        norm = np.linalg.norm(y, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-10)
        return y / norm

    iva = NaturalGradIVA(contrast_fn=contrast_fn, score_fn=score_fn)

    assert isinstance(iva, IterativeMethodBase)

    iva = NaturalGradLaplaceIVA()

    assert isinstance(iva, IterativeMethodBase)

    iva = NaturalGradGaussIVA()

    assert isinstance(iva, IterativeMethodBase)


def test_fast_iva_inheritance() -> None:
    def contrast_fn(y: np.ndarray) -> np.ndarray:
        r"""Contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.linalg.norm(y, axis=1)

    def d_contrast_fn(y) -> np.ndarray:
        r"""Derivative of contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.ones_like(y)

    def dd_contrast_fn(y) -> np.ndarray:
        r"""Second order derivative of contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.zeros_like(y)

    iva = FastIVA(
        contrast_fn=contrast_fn,
        d_contrast_fn=d_contrast_fn,
        dd_contrast_fn=dd_contrast_fn,
    )

    assert isinstance(iva, IterativeMethodBase)


def test_faster_iva_inheritance() -> None:
    def contrast_fn(y: np.ndarray) -> np.ndarray:
        r"""Contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.linalg.norm(y, axis=1)

    def d_contrast_fn(y) -> np.ndarray:
        r"""Derivative of contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.ones_like(y)

    iva = FasterIVA(contrast_fn=contrast_fn, d_contrast_fn=d_contrast_fn)

    assert isinstance(iva, IterativeMethodBase)


def test_aux_iva_inheritance() -> None:
    def contrast_fn(y: np.ndarray) -> np.ndarray:
        r"""Contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.linalg.norm(y, axis=1)

    def d_contrast_fn(y) -> np.ndarray:
        r"""Derivative of contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_frames).

        Returns:
            np.ndarray:
                The shape is (n_sources, n_frames).
        """
        return 2 * np.ones_like(y)

    iva = AuxIVA(
        contrast_fn=contrast_fn,
        d_contrast_fn=d_contrast_fn,
    )

    assert isinstance(iva, IterativeMethodBase)

    iva = AuxLaplaceIVA()

    assert isinstance(iva, IterativeMethodBase)

    iva = AuxGaussIVA()

    assert isinstance(iva, IterativeMethodBase)


def test_pds_iva_inheritance() -> None:
    iva = PDSIVA(
        contrast_fn=None,
        prox_penalty=None,
    )

    assert isinstance(iva, IterativeMethodBase)


def test_ilrma_inheritance() -> None:
    n_basis = 2

    ilrma = GaussILRMA(n_basis=n_basis)

    assert isinstance(ilrma, IterativeMethodBase)

    ilrma = TILRMA(n_basis=n_basis, dof=1000)

    assert isinstance(ilrma, IterativeMethodBase)

    ilrma = GGDILRMA(n_basis=n_basis, beta=1.95)

    assert isinstance(ilrma, IterativeMethodBase)


def test_ipsdta_inheritance() -> None:
    n_basis = 2
    n_blocks = 2

    ipsdta = GaussIPSDTA(n_basis=n_basis, n_blocks=n_blocks)

    assert isinstance(ipsdta, IterativeMethodBase)

    ipsdta = TIPSDTA(n_basis=n_basis, n_blocks=n_blocks, dof=1000)

    assert isinstance(ipsdta, IterativeMethodBase)


def test_mnmf_inheritance() -> None:
    n_basis = 2

    mnmf = GaussMNMF(n_basis=n_basis)

    assert isinstance(mnmf, IterativeMethodBase)

    mnmf = FastGaussMNMF(n_basis=n_basis)

    assert isinstance(mnmf, IterativeMethodBase)


def test_pdsbss_inheritance() -> None:
    def contrast_fn(y: np.ndarray) -> np.ndarray:
        r"""Contrast function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).

        Returns:
            np.ndarray of the shape is (n_sources, n_frames).
        """
        return 2 * np.linalg.norm(y, axis=1)

    def penalty_fn(y: np.ndarray) -> float:
        loss = contrast_fn(y)
        loss = np.sum(loss.mean(axis=-1))
        return loss

    def prox_penalty(y: np.ndarray, step_size: float = 1) -> np.ndarray:
        r"""Proximal operator of penalty function.

        Args:
            y (np.ndarray):
                The shape is (n_sources, n_bins, n_frames).
            step_size (float):
                Step size. Default: 1.

        Returns:
            np.ndarray of the shape is (n_sources, n_bins, n_frames).
        """
        norm = np.linalg.norm(y, axis=1, keepdims=True)
        return y * np.maximum(1 - step_size / norm, 0)

    pdsbss = PDSBSS(penalty_fn=penalty_fn, prox_penalty=prox_penalty)

    assert isinstance(pdsbss, IterativeMethodBase)


def test_cacgmm_inheritance() -> None:
    cacgmm = CACGMM()

    assert isinstance(cacgmm, IterativeMethodBase)
