import numpy as np

from .sisec2010 import download as download_sisec2010
from .mird import download as download_mird

__all__ = ["download_sample_speech_data"]

sisec2010_tags = ["dev1_female3", "dev1_female4"]


def download_sample_speech_data(
    sisec2010_root: str = ".data/SiSEC2010",
    mird_root: str = ".data/MIRD",
    n_sources: int = 3,
    sisec2010_tag: str = "dev1_female3",
    max_samples: int = 160000,
    conv: bool = True,
) -> np.ndarray:
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
            Choose ""dev1_female3"" or ""dev1_female4"".
            Default:""dev1_female3"".
        max_samples (int):
            Maximum time samples. Default: ``160000``.
        conv (bool):
            Convolutive mixture or not. Defalt: ``True``.

    Returns:
        numpy.ndarray:
            Source images with shape of (n_channels, n_sources, n_samples).
    """
    assert sisec2010_tag in sisec2010_tags, "Choose sisec2010_tag from {}".format(sisec2010_tags)

    sisec2010_npz_path = download_sisec2010(
        root=sisec2010_root, n_sources=n_sources, tag=sisec2010_tag
    )
    sisec2010_npz = np.load(sisec2010_npz_path)

    if conv:
        mird_npz_path = download_mird(root=mird_root, n_sources=n_sources)
        mird_npz = np.load(mird_npz_path)

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

    return waveform_src_img
