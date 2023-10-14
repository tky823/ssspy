import sys
from os import makedirs
from os.path import dirname, join, realpath

import numpy as np
import pytest

from ssspy.bss.ipsdta import TIPSDTA, GaussIPSDTA

ssspy_tests_dir = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(ssspy_tests_dir)

from dummy.utils.dataset import load_regression_data  # noqa: E402

ipsdta_root = join(ssspy_tests_dir, "mock", "regression", "bss", "ipsdta")

parameters_spatial_algorithm = ["VCD"]
parameters_source_algorithm = ["EM", "MM"]


@pytest.mark.parametrize("spatial_algorithm", parameters_spatial_algorithm)
@pytest.mark.parametrize("source_algorithm", parameters_source_algorithm)
def test_gauss_ipsdta(spatial_algorithm: str, source_algorithm: str, save_feature: bool = False):
    if source_algorithm == "EM":
        pytest.skip(reason="EM is not supported for GaussIPSDTA.")

    rng = np.random.default_rng(0)
    root = join(ipsdta_root, "gauss_ipsdta", spatial_algorithm, source_algorithm)

    if save_feature:
        (npz_input,) = load_regression_data(root=root, filenames=["input.npz"])
        spectrogram_tgt = None
        n_basis = 2
        n_iter = 10
    else:
        npz_input, npz_target = load_regression_data(root, filenames=["input.npz", "target.npz"])
        spectrogram_tgt = npz_target["spectrogram"]
        n_basis = npz_target["n_basis"].item()
        n_iter = npz_target["n_iter"].item()

    spectrogram_mix = npz_input["spectrogram"]

    if save_feature:
        n_blocks = spectrogram_mix.shape[1] // 2
        n_sources, n_bins, n_frames = spectrogram_mix.shape

        n_neighbors = n_bins // n_blocks
        n_remains = n_bins % n_blocks

        eye = np.eye(n_neighbors, dtype=np.complex128)
        rand = rng.random((n_sources, n_basis, n_blocks - n_remains, n_neighbors))
        T = rand[..., np.newaxis] * eye

        if n_remains > 0:
            eye = np.eye(n_neighbors + 1, dtype=np.complex128)
            rand = rng.random((n_sources, n_basis, n_remains, n_neighbors + 1))
            T_high = rand[..., np.newaxis] * eye

            T = T, T_high

        V = rng.random((n_sources, n_basis, n_frames))

        basis = T
        activation = V
    else:
        n_blocks = npz_target["n_blocks"].item()

        if "basis" in npz_target.keys():
            basis = npz_target["basis"]
        else:
            basis_low = npz_target["basis_low"]
            basis_high = npz_target["basis_high"]
            basis = basis_low, basis_high

        activation = npz_target["activation"]

    ipsdta = GaussIPSDTA(
        n_basis=n_basis,
        n_blocks=n_blocks,
        spatial_algorithm=spatial_algorithm,
        source_algorithm=source_algorithm,
        rng=rng,
    )
    spectrogram_est = ipsdta(
        spectrogram_mix,
        n_iter=n_iter,
        basis=basis,
        activation=activation,
    )

    if isinstance(basis, tuple):
        basis_low, basis_high = basis
        basis = {
            "basis_low": basis_low,
            "basis_high": basis_high,
        }
    else:
        basis = {
            "basis": basis,
        }

    if save_feature:
        makedirs(root, exist_ok=True)
        np.savez(
            join(root, "target.npz"),
            spectrogram=spectrogram_est,
            **basis,
            activation=activation,
            n_basis=n_basis,
            n_blocks=n_blocks,
            n_iter=n_iter,
        )
    else:
        assert np.allclose(spectrogram_est, spectrogram_tgt, atol=1e-7), np.max(
            np.abs(spectrogram_est - spectrogram_tgt)
        )


@pytest.mark.parametrize("spatial_algorithm", parameters_spatial_algorithm)
@pytest.mark.parametrize("source_algorithm", parameters_source_algorithm)
def test_t_ipsdta(spatial_algorithm: str, source_algorithm: str, save_feature: bool = False):
    if source_algorithm == "EM":
        pytest.skip(reason="EM is not supported for TIPSDTA.")

    rng = np.random.default_rng(0)
    root = join(ipsdta_root, "t_ipsdta", spatial_algorithm, source_algorithm)

    if save_feature:
        (npz_input,) = load_regression_data(root=root, filenames=["input.npz"])
        spectrogram_tgt = None
        n_basis = 2
        dof = 1000
        n_iter = 10
    else:
        npz_input, npz_target = load_regression_data(root, filenames=["input.npz", "target.npz"])
        spectrogram_tgt = npz_target["spectrogram"]
        n_basis = npz_target["n_basis"].item()
        dof = npz_target["dof"].item()
        n_iter = npz_target["n_iter"].item()

    spectrogram_mix = npz_input["spectrogram"]

    if save_feature:
        n_blocks = spectrogram_mix.shape[1] // 2
        n_sources, n_bins, n_frames = spectrogram_mix.shape

        n_neighbors = n_bins // n_blocks
        n_remains = n_bins % n_blocks

        eye = np.eye(n_neighbors, dtype=np.complex128)
        rand = rng.random((n_sources, n_basis, n_blocks - n_remains, n_neighbors))
        T = rand[..., np.newaxis] * eye

        if n_remains > 0:
            eye = np.eye(n_neighbors + 1, dtype=np.complex128)
            rand = rng.random((n_sources, n_basis, n_remains, n_neighbors + 1))
            T_high = rand[..., np.newaxis] * eye

            T = T, T_high

        V = rng.random((n_sources, n_basis, n_frames))

        basis = T
        activation = V
    else:
        n_blocks = npz_target["n_blocks"].item()

        if "basis" in npz_target.keys():
            basis = npz_target["basis"]
        else:
            basis_low = npz_target["basis_low"]
            basis_high = npz_target["basis_high"]
            basis = basis_low, basis_high

        activation = npz_target["activation"]

    ipsdta = TIPSDTA(
        n_basis=n_basis,
        n_blocks=n_blocks,
        dof=dof,
        spatial_algorithm=spatial_algorithm,
        source_algorithm=source_algorithm,
        rng=rng,
    )
    spectrogram_est = ipsdta(
        spectrogram_mix,
        n_iter=n_iter,
        basis=basis,
        activation=activation,
    )

    if isinstance(basis, tuple):
        basis_low, basis_high = basis
        basis = {
            "basis_low": basis_low,
            "basis_high": basis_high,
        }
    else:
        basis = {
            "basis": basis,
        }

    if save_feature:
        makedirs(root, exist_ok=True)
        np.savez(
            join(root, "target.npz"),
            spectrogram=spectrogram_est,
            **basis,
            activation=activation,
            n_basis=n_basis,
            n_blocks=n_blocks,
            dof=dof,
            n_iter=n_iter,
        )
    else:
        assert np.allclose(spectrogram_est, spectrogram_tgt, atol=1e-7), np.max(
            np.abs(spectrogram_est - spectrogram_tgt)
        )


def save_all_features() -> None:
    for spatial_algorithm in parameters_spatial_algorithm:
        for source_algorithm in parameters_source_algorithm:
            if source_algorithm == "EM":
                continue

            test_gauss_ipsdta(
                spatial_algorithm=spatial_algorithm,
                source_algorithm=source_algorithm,
                save_feature=True,
            )

    for spatial_algorithm in parameters_spatial_algorithm:
        for source_algorithm in parameters_source_algorithm:
            if source_algorithm == "EM":
                continue

            test_t_ipsdta(
                spatial_algorithm=spatial_algorithm,
                source_algorithm=source_algorithm,
                save_feature=True,
            )


if __name__ == "__main__":
    save_all_features()
