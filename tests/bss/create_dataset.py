import os

import numpy as np
import soundfile as sf
from scipy.io import loadmat
import scipy.signal as ss


root = "./tests/.data"
sisec2011_root = "./tests/.data/SiSEC2011"
mird_root = "./tests/.data/MIRD"
tag = "dev1_female3"


def create_sisec2011_dataset(n_sources=3) -> str:
    source_paths = []

    for src_idx in range(3):
        source_paths.append(os.path.join(sisec2011_root, "{}_src_{}.wav".format(tag, src_idx + 1)))

    channels = [3, 4, 2, 5]
    sample_rate = 16000

    source_paths = source_paths[:n_sources]
    channels = channels[:n_sources]

    n_channels = len(channels)
    npz_path = os.path.join(root, "./SiSEC2011-{}ch.npz".format(n_channels))

    assert n_channels == n_sources, "Mixing system should be determined."

    if not os.path.exists(npz_path):
        dry_sources = {}

        for src_idx, source_path in enumerate(source_paths):
            waveform_src, _ = sf.read(source_path)
            dry_sources["src_{}".format(src_idx + 1)] = waveform_src

        os.makedirs(root, exist_ok=True)
        np.savez(
            npz_path,
            sample_rate=sample_rate,
            n_sources=n_sources,
            n_channels=n_channels,
            **dry_sources
        )

    return npz_path


def create_sisec2011_mird_dataset(n_sources=3) -> str:
    source_paths = []

    for src_idx in range(3):
        source_paths.append(os.path.join(sisec2011_root, "{}_src_{}.wav".format(tag, src_idx + 1)))

    channels = [3, 4, 2, 5]
    sample_rate = 16000

    source_paths = source_paths[:n_sources]
    channels = channels[:n_sources]

    n_channels = len(channels)
    npz_path = os.path.join(root, "./SiSEC2011-inst-{}ch.npz".format(n_channels))

    assert n_channels == n_sources, "Mixing system should be determined."

    if not os.path.exists(npz_path):
        mixing = np.random.randn(n_channels, n_sources)

        dry_sources = {}

        for src_idx, source_path in enumerate(source_paths):
            coeff = mixing[:, src_idx]
            source, _ = sf.read(source_path)
            dry_sources["src_{}".format(src_idx + 1)] = coeff[:, np.newaxis] * source

        os.makedirs(root, exist_ok=True)
        np.savez(
            npz_path,
            sample_rate=sample_rate,
            n_sources=n_sources,
            n_channels=n_channels,
            **dry_sources
        )

    return npz_path


def create_mird_dataset(n_sources=3) -> str:
    degrees = [30, 345, 0, 60, 315]
    channels = [3, 4, 2, 5, 1, 6, 0, 7]
    sample_rate = 16000
    duration = 0.160

    degrees = degrees[:n_sources]
    channels = channels[:n_sources]

    n_channels = len(channels)
    n_samples = int(sample_rate * duration)

    template_rir_name = "Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_{:.3f}s)_3-3-3-8-3-3-3_1m_{:03d}.mat"  # noqa: E501
    npz_path = os.path.join(root, "./MIRD-{}ch.npz".format(n_channels))

    assert n_channels == n_sources, "Mixing system should be determined."

    if not os.path.exists(npz_path):
        rirs = {}

        for src_idx, degree in enumerate(degrees):
            rir_path = os.path.join(mird_root, template_rir_name.format(duration, degree))
            rir = _resample_mird_rir(rir_path, sample_rate_out=sample_rate)
            rirs["src_{}".format(src_idx + 1)] = rir[channels, :n_samples]

        os.makedirs(root, exist_ok=True)
        np.savez(
            npz_path, sample_rate=sample_rate, n_sources=n_sources, n_channels=n_channels, **rirs
        )

    return npz_path


def _resample_mird_rir(rir_path: str, sample_rate_out: int) -> np.ndarray:
    sample_rate_in = 48000
    rir_mat = loadmat(rir_path)
    rir = rir_mat["impulse_response"]

    rir_resampled = ss.resample_poly(rir, sample_rate_out, sample_rate_in, axis=0)

    return rir_resampled.T
