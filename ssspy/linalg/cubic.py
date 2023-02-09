import numpy as np


def cbrt(x):
    if np.iscomplexobj(x):
        amplitude = np.abs(x)
        phase = np.angle(x)
        x_cbrt = np.cbrt(amplitude) * np.exp(1j * phase / 3)
    else:
        x_cbrt = np.cbrt(x)

    return x_cbrt
