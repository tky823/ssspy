import numpy as np

from ssspy.bss.proxbss import ProxBSSBase


def contrast_fn(y: np.ndarray) -> np.ndarray:
    r"""Contrast function.

    Args:
        y (np.ndarray):
            The shape is (n_sources, n_bins, n_frames).

    Returns:
        np.ndarray of the shape is (n_sources, n_frames).
    """
    return 2 * np.linalg.norm(y, axis=1)


def penalty_fn(y: np.ndarray) -> float:
    loss = contrast_fn(y)
    loss = np.sum(loss.mean(axis=-1))
    return loss


def prox_penalty(y: np.ndarray, step_size: float = 1) -> np.ndarray:
    r"""Proximal operator of penalty function.

    Args:
        y (np.ndarray):
            The shape is (n_sources, n_bins, n_frames).
        step_size (float):
            Step size. Default: 1.

    Returns:
        np.ndarray of the shape is (n_sources, n_bins, n_frames).
    """
    norm = np.linalg.norm(y, axis=1, keepdims=True)
    return y * np.maximum(1 - step_size / norm, 0)


def test_proxbss_base() -> None:
    proxbss = ProxBSSBase(penalty_fn=penalty_fn, prox_penalty=prox_penalty)

    print(proxbss)
