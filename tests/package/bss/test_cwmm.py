import numpy as np
from scipy.special import hyp1f1

from ssspy.bss.cwmm import _kummer


def test_kummer() -> None:
    rng = np.random.default_rng(0)

    n_sources = 5
    kappa = rng.random((n_sources,))

    kummer_np = _kummer(kappa)
    kummer_sp = hyp1f1(1, n_sources, kappa)

    assert np.allclose(kummer_np, kummer_sp)
