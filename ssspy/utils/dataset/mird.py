import os
import shutil
import urllib.request

import numpy as np

reverb_durations = [0.16, 0.36, 0.61]


def download(root: str = ".data/MIRD", n_sources: int = 3, reverb_duration: float = 0.16) -> str:
    assert reverb_duration in reverb_durations, "reverb_duration should be chosen from {}.".format(
        reverb_durations
    )

    filename = (
        "Impulse_response_Acoustic_Lab_Bar-Ilan_University__"
        "Reverberation_{reverb_duration:.3f}s__3-3-3-8-3-3-3.zip"
    )
    filename = filename.format(reverb_duration=reverb_duration)
    url = (
        "https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/"
        "forschung/tools-downloads/{filename}"
    )
    url = url.format(filename=filename)
    zip_path = os.path.join(root, filename)

    degrees = [30, 345, 0, 60, 315]
    channels = [3, 4, 2, 5, 1, 6, 0, 7]
    sample_rate = 16000
    duration = reverb_duration

    degrees = degrees[:n_sources]
    channels = channels[:n_sources]

    n_channels = len(channels)
    n_samples = int(sample_rate * duration)

    template_rir_name = (
        "Impulse_response_Acoustic_Lab_Bar-Ilan_University_"
        "(Reverberation_{:.3f}s)_3-3-3-8-3-3-3_1m_{:03d}.mat"
    )

    os.makedirs(root, exist_ok=True)

    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    rir_path = os.path.join(root, template_rir_name.format(reverb_duration, 0))

    if not os.path.exists(rir_path):
        shutil.unpack_archive(zip_path, root)

    npz_path = os.path.join(root, "MIRD-{}ch.npz".format(n_channels))

    assert n_channels == n_sources, "Mixing system should be determined."

    if not os.path.exists(npz_path):
        rirs = {}

        for src_idx, degree in enumerate(degrees):
            rir_path = os.path.join(root, template_rir_name.format(duration, degree))
            rir = resample_mird_rir(rir_path, sample_rate_out=sample_rate)
            rirs["src_{}".format(src_idx + 1)] = rir[channels, :n_samples]

        np.savez(
            npz_path, sample_rate=sample_rate, n_sources=n_sources, n_channels=n_channels, **rirs
        )

    return npz_path


def resample_mird_rir(rir_path: str, sample_rate_out: int) -> np.ndarray:
    import scipy.signal as ss
    from scipy.io import loadmat

    sample_rate_in = 48000
    rir_mat = loadmat(rir_path)
    rir = rir_mat["impulse_response"]

    rir_resampled = ss.resample_poly(rir, sample_rate_out, sample_rate_in, axis=0)

    return rir_resampled.T
