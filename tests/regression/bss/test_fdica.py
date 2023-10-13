import sys
from os import makedirs
from os.path import dirname, join, realpath

import numpy as np
import pytest

from ssspy.bss.fdica import AuxLaplaceFDICA, GradLaplaceFDICA, NaturalGradLaplaceFDICA

ssspy_tests_dir = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(ssspy_tests_dir)


from dummy.utils.dataset import load_regression_data  # noqa: E402

fdica_root = join(ssspy_tests_dir, "mock", "regression", "bss", "fdica")
n_sources = 2

parameters_is_holonomic = [True, False]
parameters_spatial_algorithm = ["IP1", "IP2"]


@pytest.mark.parametrize("is_holonomic", parameters_is_holonomic)
def test_grad_laplace_fdica(is_holonomic: bool, save_feature: bool = False):
    if is_holonomic:
        root = join(fdica_root, "grad_laplace_fdica", "holonomic")
    else:
        root = join(fdica_root, "grad_laplace_fdica", "nonholonomic")

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

    fdica = GradLaplaceFDICA(is_holonomic=is_holonomic)
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

    if save_feature:
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
def test_natural_grad_laplace_fdica(is_holonomic: bool, save_feature: bool = False):
    if is_holonomic:
        root = join(fdica_root, "natural_grad_laplace_fdica", "holonomic")
    else:
        root = join(fdica_root, "natural_grad_laplace_fdica", "nonholonomic")

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

    fdica = NaturalGradLaplaceFDICA(is_holonomic=is_holonomic)
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

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
def test_aux_laplace_fdica(spatial_algorithm: str, save_feature: bool = False):
    root = join(fdica_root, "aux_laplace_fdica", spatial_algorithm)

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

    fdica = AuxLaplaceFDICA(spatial_algorithm=spatial_algorithm)
    spectrogram_est = fdica(spectrogram_mix, n_iter=n_iter)

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
        test_grad_laplace_fdica(is_holonomic=is_holonomic, save_feature=True)

    for is_holonomic in parameters_is_holonomic:
        test_natural_grad_laplace_fdica(is_holonomic=is_holonomic, save_feature=True)

    for spatial_algorithm in parameters_spatial_algorithm:
        test_aux_laplace_fdica(spatial_algorithm=spatial_algorithm, save_feature=True)


if __name__ == "__main__":
    save_all_features()
