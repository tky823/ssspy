import os

import numpy as np
import soundfile as sf


root = "./tests/.data"
sisec2011_root = "./tests/.data/SiSEC2011"
tag = "dev1_female3"


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
