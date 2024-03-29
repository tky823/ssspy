import hashlib
import json
import os
import urllib.request
import warnings
from typing import List, Optional, Tuple

import numpy as np

from ssspy.utils.dataset import download_sample_speech_data as _download


def download_sample_speech_data(
    sisec2010_root: str = "./tests/.data/SiSEC2010",
    mird_root: str = "./tests/.data/MIRD",
    n_sources: int = 3,
    sisec2010_tag: str = "dev1_female3",
    max_duration: float = 10,
    conv: bool = True,
    cache_dir: str = ".tests/.data/.cache",
) -> Tuple[np.ndarray, int]:
    hash = hashlib.sha256(sisec2010_root.encode("utf-8")).hexdigest()
    hash += hashlib.sha256(mird_root.encode("utf-8")).hexdigest()
    hash += hashlib.sha256(str(n_sources).encode("utf-8")).hexdigest()
    hash += hashlib.sha256(sisec2010_tag.encode("utf-8")).hexdigest()
    hash += hashlib.sha256(str(max_duration).encode("utf-8")).hexdigest()
    hash += hashlib.sha256(str(conv).encode("utf-8")).hexdigest()

    # because concatenated hash is too long
    hash = hashlib.sha256(hash.encode("utf-8")).hexdigest()

    npz_path = os.path.join(cache_dir, "{}.npz".format(hash))

    if os.path.exists(npz_path):
        npz = np.load(npz_path)
        waveform_src_img, sample_rate = npz["waveform_src_img"], npz["sample_rate"]
        sample_rate = sample_rate.item()
    else:
        waveform_src_img, sample_rate = _download(
            sisec2010_root=sisec2010_root,
            mird_root=mird_root,
            n_sources=n_sources,
            sisec2010_tag=sisec2010_tag,
            max_duration=max_duration,
            conv=conv,
        )
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(npz_path, waveform_src_img=waveform_src_img, sample_rate=sample_rate)

    return waveform_src_img, sample_rate


def download_ssspy_data(path: str, filename: Optional[str] = None, branch: str = "main") -> None:
    """Download file from https://github.com/tky823/ssspy-data.

    Args:
        path (str): Path to file in https://github.com/tky823/ssspy-data.
        filename (str, optional): File name to save data. If ``None``,
            base name of ``path`` is used.
        branch (str, optional): Branch name of https://github.com/tky823/ssspy-data.

    """
    url = f"https://github.com/tky823/ssspy-data/raw/{branch}/{path}"

    if filename is None:
        filename = os.path.basename(url)

    root = os.path.dirname(filename)

    if root:
        os.makedirs(root, exist_ok=True)

    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)


def load_regression_data(root: str, filenames: Optional[List[str]] = None) -> Tuple:
    """Load regression data.

    Args:
        root (str): Root to save regression data, where url.json is placed.
        filenames (str, optional): Filenames to download.

    Returns:
        tuple: Tuple containing data of specified filenames.

    """
    url_json_path = os.path.join(root, "url.json")

    with open(url_json_path) as f:
        urls = json.load(f)

    if filenames is None:
        warnings.warn("It is recommended to specify filenames to ensure order.", UserWarning)

        filenames = []

        for file in urls["files"]:
            filename = file["filename"]
            filenames.append(filename)

    npz = {}

    for file in urls["files"]:
        filename = file["filename"]
        location = file["location"]

        if filename not in filenames:
            continue

        path = os.path.join(root, filename)

        download_ssspy_data(location, path)

        npz[filename] = np.load(path)

    sorted_npz = []

    for filename in filenames:
        sorted_npz.append(npz[filename])

    return tuple(sorted_npz)
