import os
import shutil
import urllib.request

import numpy as np

from ...io import wavread


def download(root: str = ".data/SiSEC2010", n_sources: int = 3, tag: str = "dev1_female3") -> str:
    filename = "dev1.zip"
    url = "http://www.irisa.fr/metiss/SiSEC10/underdetermined/{}".format(filename)
    zip_path = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    if not os.path.exists(os.path.join(root, "{}_inst_matrix.mat".format(tag))):
        shutil.unpack_archive(zip_path, root)

    source_paths = []

    for src_idx in range(n_sources):
        source_path = os.path.join(root, "{}_src_{}.wav".format(tag, src_idx + 1))
        source_paths.append(source_path)

    channels = [3, 4, 2, 5]
    sample_rate = 16000

    source_paths = source_paths[:n_sources]
    channels = channels[:n_sources]

    n_channels = len(channels)
    npz_path = os.path.join(root, "SiSEC2010-{}ch.npz".format(n_channels))

    assert n_channels == n_sources, "Mixing system should be determined."

    if not os.path.exists(npz_path):
        dry_sources = {}

        for src_idx, source_path in enumerate(source_paths):
            data, _ = wavread(source_path, return_2d=False)
            dry_sources["src_{}".format(src_idx + 1)] = data

        np.savez(
            npz_path,
            sample_rate=sample_rate,
            n_sources=n_sources,
            n_channels=n_channels,
            **dry_sources,
        )

    return npz_path
