import sys
from os import makedirs
from os.path import dirname, join, realpath

import numpy as np
import pytest

from ssspy.bss.ilrma import GGDILRMA, TILRMA, GaussILRMA

ssspy_tests_dir = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(ssspy_tests_dir)

from dummy.utils.dataset import load_regression_data  # noqa: E402

ilrma_root = join(ssspy_tests_dir, "mock", "regression", "bss", "ilrma")

parameters_spatial_algorithm = ["IP1", "IP2", "ISS1", "ISS2", "IPA"]
parameters_source_algorithm = ["MM", "ME"]


@pytest.mark.parametrize("spatial_algorithm", parameters_spatial_algorithm)
@pytest.mark.parametrize("source_algorithm", parameters_source_algorithm)
def test_gauss_ilrma(spatial_algorithm: str, source_algorithm: str, save_feature: bool = False):
    root = join(ilrma_root, "gauss_ilrma", spatial_algorithm, source_algorithm)

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

    ilrma = GaussILRMA(
        n_basis=n_basis,
        spatial_algorithm=spatial_algorithm,
        source_algorithm=source_algorithm,
    )
    spectrogram_est = ilrma(spectrogram_mix, n_iter=n_iter)

    if save_feature:
        makedirs(root, exist_ok=True)
        np.savez(
            join(root, "target.npz"),
            spectrogram=spectrogram_est,
            n_basis=n_basis,
            n_iter=n_iter,
        )
    else:
        assert np.allclose(spectrogram_est, spectrogram_tgt, atol=1e-7), np.max(
            np.abs(spectrogram_est - spectrogram_tgt)
        )


@pytest.mark.parametrize("spatial_algorithm", parameters_spatial_algorithm)
@pytest.mark.parametrize("source_algorithm", parameters_source_algorithm)
def test_t_ilrma(spatial_algorithm: str, source_algorithm: str, save_feature: bool = False):
    if spatial_algorithm == "IPA":
        pytest.skip(reason="IPA is not supported for TILRMA.")

    root = join(ilrma_root, "t_ilrma", spatial_algorithm, source_algorithm)

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

    ilrma = TILRMA(
        n_basis=n_basis,
        dof=dof,
        spatial_algorithm=spatial_algorithm,
        source_algorithm=source_algorithm,
    )
    spectrogram_est = ilrma(spectrogram_mix, n_iter=n_iter)

    if save_feature:
        makedirs(root, exist_ok=True)
        np.savez(
            join(root, "target.npz"),
            spectrogram=spectrogram_est,
            n_basis=n_basis,
            dof=dof,
            n_iter=n_iter,
        )
    else:
        assert np.allclose(spectrogram_est, spectrogram_tgt, atol=1e-7), np.max(
            np.abs(spectrogram_est - spectrogram_tgt)
        )


@pytest.mark.parametrize("spatial_algorithm", parameters_spatial_algorithm)
@pytest.mark.parametrize("source_algorithm", parameters_source_algorithm)
def test_ggd_ilrma(spatial_algorithm: str, source_algorithm: str, save_feature: bool = False):
    if spatial_algorithm == "IPA":
        pytest.skip(reason="IPA is not supported for GGDILRMA.")

    if source_algorithm == "ME":
        pytest.skip(reason="ME is not supported for GGDILRMA.")

    root = join(ilrma_root, "ggd_ilrma", spatial_algorithm, source_algorithm)

    if save_feature:
        (npz_input,) = load_regression_data(root=root, filenames=["input.npz"])
        spectrogram_tgt = None
        n_basis = 2
        beta = 1.5
        n_iter = 10
    else:
        npz_input, npz_target = load_regression_data(root, filenames=["input.npz", "target.npz"])
        spectrogram_tgt = npz_target["spectrogram"]
        n_basis = npz_target["n_basis"].item()
        beta = npz_target["beta"].item()
        n_iter = npz_target["n_iter"].item()

    spectrogram_mix = npz_input["spectrogram"]

    ilrma = GGDILRMA(
        n_basis=n_basis,
        beta=beta,
        spatial_algorithm=spatial_algorithm,
        source_algorithm=source_algorithm,
    )
    spectrogram_est = ilrma(spectrogram_mix, n_iter=n_iter)

    if save_feature:
        makedirs(root, exist_ok=True)
        np.savez(
            join(root, "target.npz"),
            spectrogram=spectrogram_est,
            n_basis=n_basis,
            beta=beta,
            n_iter=n_iter,
        )
    else:
        assert np.allclose(spectrogram_est, spectrogram_tgt, atol=1e-7), np.max(
            np.abs(spectrogram_est - spectrogram_tgt)
        )


def save_all_features() -> None:
    for spatial_algorithm in parameters_spatial_algorithm:
        for source_algorithm in parameters_source_algorithm:
            test_gauss_ilrma(
                spatial_algorithm=spatial_algorithm,
                source_algorithm=source_algorithm,
                save_feature=True,
            )

    for spatial_algorithm in parameters_spatial_algorithm:
        if spatial_algorithm == "IPA":
            continue

        for source_algorithm in parameters_source_algorithm:
            test_t_ilrma(
                spatial_algorithm=spatial_algorithm,
                source_algorithm=source_algorithm,
                save_feature=True,
            )

    for spatial_algorithm in parameters_spatial_algorithm:
        if spatial_algorithm == "IPA":
            continue

        for source_algorithm in parameters_source_algorithm:
            if source_algorithm == "ME":
                continue

            test_ggd_ilrma(
                spatial_algorithm=spatial_algorithm,
                source_algorithm=source_algorithm,
                save_feature=True,
            )


if __name__ == "__main__":
    save_all_features()
