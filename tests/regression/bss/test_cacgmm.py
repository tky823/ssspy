import sys
from os import makedirs
from os.path import dirname, join, realpath

import numpy as np

from ssspy.bss.cacgmm import CACGMM

ssspy_tests_dir = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(ssspy_tests_dir)

from dummy.utils.dataset import load_regression_data  # noqa: E402

cacgmm_root = join(ssspy_tests_dir, "mock", "regression", "bss", "cacgmm")


def test_cacgmm(save_feature: bool = False):
    rng = np.random.default_rng(0)

    if save_feature:
        (npz_input,) = load_regression_data(root=cacgmm_root, filenames=["input.npz"])
        spectrogram_tgt = None
        n_iter = 10
    else:
        npz_input, npz_target = load_regression_data(
            root=cacgmm_root, filenames=["input.npz", "target.npz"]
        )
        spectrogram_tgt = npz_target["spectrogram"]
        n_iter = npz_target["n_iter"].item()

    spectrogram_mix = npz_input["spectrogram"]

    cacgmm = CACGMM(rng=rng)
    spectrogram_est = cacgmm(spectrogram_mix, n_iter=n_iter)

    if save_feature:
        makedirs(cacgmm_root, exist_ok=True)
        np.savez(
            join(cacgmm_root, "target.npz"),
            spectrogram=spectrogram_est,
            n_iter=n_iter,
        )
    else:
        assert np.allclose(spectrogram_est, spectrogram_tgt, atol=1e-7), np.max(
            np.abs(spectrogram_est - spectrogram_tgt)
        )


def save_all_features() -> None:
    test_cacgmm(save_feature=True)


if __name__ == "__main__":
    save_all_features()
