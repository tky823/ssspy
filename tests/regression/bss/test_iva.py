import sys
from os import makedirs
from os.path import dirname, join, realpath

import numpy as np
import pytest

from ssspy.bss.iva import AuxIVA, FastIVA, GradIVA, NaturalGradIVA

ssspy_tests_dir = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(ssspy_tests_dir)

from dummy.utils.dataset import load_regression_data  # noqa: E402

iva_root = join(ssspy_tests_dir, "mock", "regression", "bss", "iva")

parameters_is_holonomic = [True, False]
parameters_spatial_algorithm = ["IP1", "IP2", "ISS1", "ISS2", "IPA"]


@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_grad_iva(is_holonomic: bool, save_feature: bool = False):
    if is_holonomic:
        root = join(iva_root, "grad_iva", "holonomic")
    else:
        root = join(iva_root, "grad_iva", "nonholonomic")

    if save_feature:
        (npz_input,) = load_regression_data(root=root, filenames=["input.npz"])
        spectrogram_tgt = None
        n_iter = 10
    else:
        npz_input, npz_target = load_regression_data(
            root=root, filenames=["input.npz", "target.npz"]
        )
        spectrogram_tgt = npz_target["spectrogram"]
        n_iter = npz_target["n_iter"].item()

    spectrogram_mix = npz_input["spectrogram"]

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

    iva = GradIVA(
        contrast_fn=contrast_fn,
        score_fn=score_fn,
        is_holonomic=is_holonomic,
    )
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    if save_feature:
        makedirs(root, exist_ok=True)
        np.savez(
            join(root, "target.npz"),
            spectrogram=spectrogram_est,
            n_iter=n_iter,
        )
    else:
        assert np.allclose(spectrogram_est, spectrogram_tgt, atol=1e-7), np.max(
            np.abs(spectrogram_est - spectrogram_tgt)
        )


@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_natural_grad_iva(is_holonomic: bool, save_feature: bool = False):
    if is_holonomic:
        root = join(iva_root, "natural_grad_iva", "holonomic")
    else:
        root = join(iva_root, "natural_grad_iva", "nonholonomic")

    if save_feature:
        (npz_input,) = load_regression_data(root=root, filenames=["input.npz"])
        spectrogram_tgt = None
        n_iter = 10
    else:
        npz_input, npz_target = load_regression_data(
            root=root, filenames=["input.npz", "target.npz"]
        )
        spectrogram_tgt = npz_target["spectrogram"]
        n_iter = npz_target["n_iter"].item()

    spectrogram_mix = npz_input["spectrogram"]

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

    iva = NaturalGradIVA(
        contrast_fn=contrast_fn,
        score_fn=score_fn,
        is_holonomic=is_holonomic,
    )
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    if save_feature:
        makedirs(root, exist_ok=True)
        np.savez(
            join(root, "target.npz"),
            spectrogram=spectrogram_est,
            n_iter=n_iter,
        )
    else:
        assert np.allclose(spectrogram_est, spectrogram_tgt, atol=1e-7), np.max(
            np.abs(spectrogram_est - spectrogram_tgt)
        )


@pytest.mark.parametrize("spatial_algorithm", parameters_spatial_algorithm)
def test_aux_iva(spatial_algorithm: str, save_feature: bool = False):
    root = join(iva_root, "aux_iva", spatial_algorithm)

    if save_feature:
        (npz_input,) = load_regression_data(root=root, filenames=["input.npz"])
        spectrogram_tgt = None
        n_iter = 10
    else:
        npz_input, npz_target = load_regression_data(
            root=root, filenames=["input.npz", "target.npz"]
        )
        spectrogram_tgt = npz_target["spectrogram"]
        n_iter = npz_target["n_iter"].item()

    spectrogram_mix = npz_input["spectrogram"]

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
        spatial_algorithm=spatial_algorithm,
        contrast_fn=contrast_fn,
        d_contrast_fn=d_contrast_fn,
    )
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    if save_feature:
        makedirs(root, exist_ok=True)
        np.savez(
            join(root, "target.npz"),
            spectrogram=spectrogram_est,
            n_iter=n_iter,
        )
    else:
        assert np.allclose(spectrogram_est, spectrogram_tgt, atol=1e-7), np.max(
            np.abs(spectrogram_est - spectrogram_tgt)
        )


def test_fast_iva(save_feature: bool = False):
    root = join(iva_root, "fast_iva")

    if save_feature:
        (npz_input,) = load_regression_data(root=root, filenames=["input.npz"])
        spectrogram_tgt = None
        n_iter = 5
    else:
        npz_input, npz_target = load_regression_data(
            root=root, filenames=["input.npz", "target.npz"]
        )
        spectrogram_tgt = npz_target["spectrogram"]
        n_iter = npz_target["n_iter"].item()

    spectrogram_mix = npz_input["spectrogram"]

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
        r"""Second roder derivative of contrast function.

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
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    if save_feature:
        makedirs(root, exist_ok=True)
        np.savez(
            join(root, "target.npz"),
            spectrogram=spectrogram_est,
            n_iter=n_iter,
        )
    else:
        assert np.allclose(spectrogram_est, spectrogram_tgt, atol=1e-7), np.max(
            np.abs(spectrogram_est - spectrogram_tgt)
        )


def save_all_features() -> None:
    for is_holonomic in parameters_is_holonomic:
        test_grad_iva(is_holonomic=is_holonomic, save_feature=True)

    for is_holonomic in parameters_is_holonomic:
        test_natural_grad_iva(is_holonomic=is_holonomic, save_feature=True)

    for spatial_algorithm in parameters_spatial_algorithm:
        test_aux_iva(spatial_algorithm=spatial_algorithm, save_feature=True)

    test_fast_iva(save_feature=True)


if __name__ == "__main__":
    save_all_features()
