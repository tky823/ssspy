from typing import Callable, List, Optional, Union

import numpy as np

from .base import IterativeMethodBase


class CWMMBase(IterativeMethodBase):
    r"""Base class of complex Watson mixture model (cWMM).

    Args:
        n_sources (int, optional):
            Number of sources to be separated.
            If ``None`` is given, ``n_sources`` is determined by number of channels
            in input spectrogram. Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        rng (numpy.random.Generator, optioinal):
            Random number generator. This is mainly used to randomly initialize parameters
            of cWMM. If ``None`` is given, ``np.random.default_rng()`` is used.
            Default: ``None``.
    """

    def __init__(
        self,
        n_sources: Optional[int] = None,
        callbacks: Optional[
            Union[
                Callable[["CWMMBase"], None],
                List[Callable[["CWMMBase"], None]],
            ]
        ] = None,
        record_loss: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(callbacks=callbacks, record_loss=record_loss)

        self.n_sources = n_sources

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

    def __call__(
        self, input: np.ndarray, n_iter: int = 100, initial_call: bool = True, **kwargs
    ) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                The number of iterations of demixing filter updates.
                Default: ``100``.
            initial_call (bool):
                If ``True``, perform callbacks (and computation of loss if necessary)
                before iterations.

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_channels, n_bins, n_frames).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        raise NotImplementedError("Implement '__call__' method.")

    def __repr__(self) -> str:
        s = "CWMM("

        if self.n_sources is not None:
            s += "n_sources={n_sources}, "

        s += "record_loss={record_loss}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(self, **kwargs) -> None:
        r"""Reset attributes by given keyword arguments.

        Args:
            kwargs:
                Keyword arguments to set as attributes of CWMM.
        """
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        X = self.input

        norm = np.linalg.norm(X, axis=0)
        Z = X / norm
        self.unit_input = Z

        n_sources = self.n_sources
        n_channels, n_bins, n_frames = X.shape

        if n_sources is None:
            n_sources = n_channels

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        self._init_parameters(rng=self.rng)

    def _init_parameters(self, rng: Optional[np.random.Generator] = None) -> None:
        r"""Initialize parameters of cWMM.

        Args:
            rng (numpy.random.Generator, optional):
                Random number generator. If ``None`` is given,
                ``np.random.default_rng()`` is used.
                Default: ``None``.
        """
        n_sources, n_channels = self.n_sources, self.n_channels

        if rng is None:
            rng = np.random.default_rng()

        alpha = rng.random((n_sources,))
        alpha = alpha / alpha.sum(axis=0)

        d = self.rng.random((n_sources, n_channels))
        kappa = self.rng.random((n_sources,))

        self.mixing = alpha
        self.mean = d
        self.concentration = kappa

        # The shape of posterior is (n_sources, n_bins, n_frames).
        # This is always required to satisfy posterior.sum(axis=0) = 1
        self.posterior: np.ndarray = None

    def separate(self, input: np.ndarray) -> np.ndarray:
        r"""Separate ``input``.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_sources, n_bins, n_frames).
        """
        raise NotImplementedError("Implement 'separate' method.")

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        Returns:
            Computed loss.
        """
        raise NotImplementedError("Implement 'compute_loss' method.")


def _kummer(kappa: np.ndarray) -> np.ndarray:
    """Kummer function M(1, N; X).

    Args:
        kappa (np.ndarray): Concentration parameter of shape (n_sources,)
            or (n_sources, n_bins).

    Returns:
        np.ndarray of computed values of same shape as kappa.

    """
    ndim = kappa.ndim

    if ndim == 1:
        kappa = kappa[:, np.newaxis]
    elif ndim == 2:
        pass
    else:
        raise ValueError("kappa should be 1D or 2D.")

    n_sources = kappa.shape[0]
    indices = np.arange(1, n_sources)
    cumprod = np.cumprod(indices, axis=0)

    kappa_l = kappa ** indices[:, np.newaxis, np.newaxis]
    scale = cumprod[-1] / kappa_l[-1]
    exp_kappa = np.exp(kappa)
    terms = kappa_l / cumprod[:, np.newaxis, np.newaxis]
    k = scale * (exp_kappa - np.sum(terms[:-1], axis=0) - 1)

    if ndim == 1:
        k = k.squeeze(axis=-1)

    return k
