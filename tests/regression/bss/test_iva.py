import sys
from os.path import dirname, join, realpath

import numpy as np

from ssspy.bss.iva import AuxIVA

ssspy_tests_dir = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(ssspy_tests_dir)

from dummy.utils.dataset import load_regression_data  # noqa: E402

iva_root = join(ssspy_tests_dir, "mock", "regression", "bss", "iva")


def test_aux_iva():
    root = join(iva_root, "aux_iva")

    npz_input, npz_target = load_regression_data(root, filenames=["input.npz", "target.npz"])
    spectrogram_mix = npz_input["spectrogram"]
    spectrogram_tgt = npz_target["spectrogram"]
    n_iter = npz_target["n_iter"].item()

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
        contrast_fn=contrast_fn, d_contrast_fn=d_contrast_fn
    )
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert np.allclose(spectrogram_est, spectrogram_tgt)
