import numpy as np


def cbrt(x: np.ndarray) -> np.ndarray:
    """Return cube-root of an array.

    Args:
        x (np.ndarray):
            Values to compute cube-root. Complex value is available.

    Returns:
        np.ndarray of cube-root.

    """
    if np.iscomplexobj(x):
        amplitude = np.abs(x)
        phase = np.angle(x)
        x_cbrt = np.cbrt(amplitude) * np.exp(1j * phase / 3)
    else:
        x_cbrt = np.cbrt(x)

    return x_cbrt
