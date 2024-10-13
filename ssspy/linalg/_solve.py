import numpy as np
from packaging import version

np_version = np.__version__

IS_NUMPY_GE_2 = version.parse(np.__version__) >= version.parse("2")


def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    requires_new_axis = IS_NUMPY_GE_2 and a.ndim == b.ndim + 1

    if requires_new_axis:
        b = b[..., np.newaxis]

    x = np.linalg.solve(a, b)

    if requires_new_axis:
        x = x[..., 0]
        b = b[..., 0]

    return x
