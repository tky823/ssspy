import sys
from os import makedirs
from os.path import dirname, join, realpath

import numpy as np
import pytest

from ssspy.bss.mnmf import FastGaussMNMF, GaussMNMF

ssspy_tests_dir = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(ssspy_tests_dir)

from dummy.utils.dataset import load_regression_data  # noqa: E402

mnmf_root = join(ssspy_tests_dir, "mock", "regression", "bss", "mnmf")

parameters_diagonalizer_algorithm = ["IP1", "IP2"]


def test_gauss_mnmf(save_feature: bool = False):
    rng = np.random.default_rng(0)
    root = join(mnmf_root, "gauss_mnmf")

    if save_feature:
        (npz_input,) = load_regression_data(root=root, filenames=["input.npz"])
        spectrogram_tgt = None
        n_basis = 2
        n_iter = 5
    else:
        npz_input, npz_target = load_regression_data(
            root=root, filenames=["input.npz", "target.npz"]
        )
        spectrogram_tgt = npz_target["spectrogram"]
        n_basis = npz_target["n_basis"].item()
        n_iter = npz_target["n_iter"].item()

    spectrogram_mix = npz_input["spectrogram"]

    n_channels, n_bins, n_frames = spectrogram_mix.shape
    n_sources = n_channels

    if save_feature:
        basis = rng.random((n_sources, n_bins, n_basis))
        activation = rng.random((n_sources, n_basis, n_frames))

        spatial = np.eye(n_channels, dtype=spectrogram_mix.dtype)
        trace = np.trace(spatial, axis1=-2, axis2=-1)
        spatial = spatial / np.real(trace)
        spatial = np.tile(spatial, reps=(n_sources, n_bins, 1, 1))
    else:
        basis = npz_target["basis"]
        activation = npz_target["activation"]
        spatial = npz_target["spatial"]

    mnmf = GaussMNMF(
        n_basis=n_basis,
        n_sources=n_sources,
        rng=rng,
    )
    spectrogram_est = mnmf(
        spectrogram_mix,
        n_iter=n_iter,
        basis=basis,
        activation=activation,
        spatial=spatial,
    )

    if save_feature:
        makedirs(root, exist_ok=True)
        np.savez(
            join(root, "target.npz"),
            spectrogram=spectrogram_est,
            basis=basis,
            activation=activation,
            n_basis=n_basis,
            spatial=spatial,
            n_iter=n_iter,
        )
    else:
        assert np.allclose(spectrogram_est, spectrogram_tgt, atol=1e-7), np.max(
            np.abs(spectrogram_est - spectrogram_tgt)
        )


@pytest.mark.parametrize("diagonalizer_algorithm", parameters_diagonalizer_algorithm)
def test_fast_gauss_mnmf(diagonalizer_algorithm: str, save_feature: bool = False):
    rng = np.random.default_rng(0)
    root = join(mnmf_root, "fast_gauss_mnmf", diagonalizer_algorithm)

    if save_feature:
        (npz_input,) = load_regression_data(root=root, filenames=["input.npz"])
        spectrogram_tgt = None
        n_basis = 2
        n_iter = 5
    else:
        npz_input, npz_target = load_regression_data(
            root=root, filenames=["input.npz", "target.npz"]
        )
        spectrogram_tgt = npz_target["spectrogram"]
        n_basis = npz_target["n_basis"].item()
        n_iter = npz_target["n_iter"].item()

    spectrogram_mix = npz_input["spectrogram"]

    n_channels, n_bins, n_frames = spectrogram_mix.shape
    n_sources = n_channels

    if save_feature:
        basis = rng.random((n_sources, n_bins, n_basis))
        activation = rng.random((n_sources, n_basis, n_frames))
        spatial = rng.random((n_bins, n_sources, n_channels))
        diagonalizer = np.eye(n_channels, dtype=np.complex128)
        diagonalizer = np.tile(diagonalizer, reps=(n_bins, 1, 1))
    else:
        basis = npz_target["basis"]
        activation = npz_target["activation"]
        spatial = npz_target["spatial"]
        diagonalizer = npz_target["diagonalizer"]

    mnmf = FastGaussMNMF(
        n_basis=n_basis,
        n_sources=n_sources,
        diagonalizer_algorithm=diagonalizer_algorithm,
        rng=rng,
    )
    spectrogram_est = mnmf(
        spectrogram_mix,
        n_iter=n_iter,
        basis=basis,
        activation=activation,
        spatial=spatial,
        diagonalizer=diagonalizer,
    )

    if save_feature:
        makedirs(root, exist_ok=True)
        np.savez(
            join(root, "target.npz"),
            spectrogram=spectrogram_est,
            basis=basis,
            activation=activation,
            spatial=spatial,
            diagonalizer=diagonalizer,
            n_basis=n_basis,
            n_iter=n_iter,
        )
    else:
        assert np.allclose(spectrogram_est, spectrogram_tgt, atol=1e-7), np.max(
            np.abs(spectrogram_est - spectrogram_tgt)
        )


def save_all_features() -> None:
    test_gauss_mnmf(save_feature=True)

    for diagonalizer_algorithm in parameters_diagonalizer_algorithm:
        test_fast_gauss_mnmf(diagonalizer_algorithm=diagonalizer_algorithm, save_feature=True)


if __name__ == "__main__":
    save_all_features()
