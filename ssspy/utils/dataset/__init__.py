import numpy as np

from .sisec2010 import download as download_sisec2010
from .mird import download as download_mird


def download_dummy_data(
    sisec2010_root: str = ".data/SiSEC2010",
    mird_root: str = ".data/MIRD",
    n_sources: int = 3,
    sisec2010_tag: str = "dev1_female3",
    max_samples: int = 160000,
):
    sisec2010_npz_path = download_sisec2010(
        root=sisec2010_root, n_sources=n_sources, tag=sisec2010_tag
    )
    mird_npz_path = download_mird(root=mird_root, n_sources=n_sources)

    sisec2010_npz = np.load(sisec2010_npz_path)
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

    return waveform_src_img
