from typing import Tuple

import numpy as np

from .mird import download as download_mird
from .sisec2010 import download as download_sisec2010

__all__ = ["download_sample_speech_data"]

sisec2010_tags = ["dev1_female3", "dev1_female4"]


def download_sample_speech_data(
    sisec2010_root: str = ".data/SiSEC2010",
    mird_root: str = ".data/MIRD",
    n_sources: int = 3,
    sisec2010_tag: str = "dev1_female3",
    max_duration: float = 10,
    reverb_duration: float = 0.16,
    conv: bool = True,
) -> Tuple[np.ndarray, int]:
    r"""Download sample speech data to test sepration methods.

    This function returns source images of sample speech data.

    Args:
        sisec2010_root (str):
            Path to save SiSEC2010 dataset. Default: ".data/SiSEC2010".
        mird_root (str):
            Path to save MIRD dataset. Default: ".data/MIRD".
        n_sources (int):
            Number of sources included in sample data.
        sisec2010_tag (str):
            Tag of SiSEC 2010 data.
            Choose ``dev1_female3`` or ``dev1_female4``.
            Default: ``dev1_female3``.
        max_duration (float):
            Maximum duration. Default: ``160000``.
        reverb_duration (float):
            Duration of reverberation in MIRD.
            Choose ``0.16``, ``0.36``, ``0.61``. Default: ``0.16``.
        conv (bool):
            Convolutive mixture or not. Defalt: ``True``.

    Returns:
        Tuple of source images and sampling rate.
        The source images is numpy.ndarry with shape of (n_channels, n_sources, n_samples).
    """
    assert sisec2010_tag in sisec2010_tags, "Choose sisec2010_tag from {}".format(sisec2010_tags)
    sample_rate = 16000  # Only 16khz is supported.
    max_samples = int(sample_rate * max_duration)

    sisec2010_npz_path = download_sisec2010(
        root=sisec2010_root, n_sources=n_sources, tag=sisec2010_tag
    )
    sisec2010_npz = np.load(sisec2010_npz_path)

    assert sample_rate == sisec2010_npz["sample_rate"].item(), "Invalid sampling rate is detected."

    if conv:
        mird_npz_path = download_mird(
            root=mird_root, n_sources=n_sources, reverb_duration=reverb_duration
        )
        mird_npz = np.load(mird_npz_path)

        assert sample_rate == mird_npz["sample_rate"].item(), "Invalid sampling rate is detected."

        waveform_src_img = []

        for src_idx in range(n_sources):
            key = "src_{}".format(src_idx + 1)
            waveform_src = sisec2010_npz[key][:max_samples]
            n_samples = len(waveform_src)
            _waveform_src_img = []

            for waveform_rir in mird_npz[key]:
                waveform_conv = np.convolve(waveform_src, waveform_rir)[:n_samples]
                _waveform_src_img.append(waveform_conv)

            _waveform_src_img = np.stack(_waveform_src_img, axis=0)  # (n_channels, n_samples)
            waveform_src_img.append(_waveform_src_img)

        waveform_src_img = np.stack(waveform_src_img, axis=1)  # (n_channels, n_sources, n_samples)
    else:
        waveform_src_img = []

        rng = np.random.default_rng(seed=42)
        mixing = rng.standard_normal((n_sources, n_sources))

        for src_idx in range(n_sources):
            key = "src_{}".format(src_idx + 1)
            _mixing = mixing[:, src_idx]
            waveform_src = sisec2010_npz[key][:max_samples]
            _waveform_src_img = _mixing[:, np.newaxis] * waveform_src
            waveform_src_img.append(_waveform_src_img)

        waveform_src_img = np.stack(waveform_src_img, axis=1)  # (n_channels, n_sources, n_samples)

    return waveform_src_img, sample_rate
