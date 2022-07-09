from typing import Optional, Union, List, Callable

import numpy as np

from ..algorithm import projection_back
from ..linalg import prox

EPS = 1e-10


class PDSBSSbase:
    r"""Base class of based on blind source separation \
    via proximal splitting algorithm [#yatabe2018determined]_.

    Args:
        penalty_fn (callable):
            Penalty function that determines source model.
        prox_penalty (callable):
            Proximal operator of penalty function.
            Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_apply_projection_back (bool):
            If ``should_apply_projection_back=True``, the projection back is applied to \
            estimated spectrograms. Default: ``True``.
        should_record_loss (bool):
            Record the loss at each iteration of the update algorithm \
            if ``should_record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.

    .. [#yatabe2018determined] K. Yatabe and D. Kitamura,
        "Determined blind source separation via proximal splitting algorithm,"
        in *Proc of ICASSP*, pp. 776-780, 2018.
    """

    def __init__(
        self,
        penalty_fn: Callable[[np.ndarray, np.ndarray], float] = None,
        prox_penalty: Callable[[np.ndarray, float], np.ndarray] = None,
        callbacks: Optional[
            Union[Callable[["PDSBSSbase"], None], List[Callable[["PDSBSSbase"], None]]]
        ] = None,
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        if penalty_fn is None:
            raise ValueError("Specify penalty function.")
        else:
            self.penalty_fn = penalty_fn

        if prox_penalty is None:
            raise ValueError("Specify proximal operator of penalty function.")
        else:
            self.prox_penalty = prox_penalty

        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None

        self.input = None
        self.should_apply_projection_back = should_apply_projection_back

        if reference_id is None and should_apply_projection_back:
            raise ValueError("Specify 'reference_id' if should_apply_projection_back=True.")
        else:
            self.reference_id = reference_id

        self.should_record_loss = should_record_loss

        if self.should_record_loss:
            self.loss = []
        else:
            self.loss = None

    def __call__(self, input: np.ndarray, n_iter: int = 100, **kwargs) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                Mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                Number of iterations of demixing filter updates.
                Default: 100.

        Returns:
            numpy.ndarray:
                The separated signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        raise NotImplementedError("Implement '__call__' method.")

    def __repr__(self) -> str:
        s = "PDSBSS("
        s += "should_apply_projection_back={should_apply_projection_back}"
        s += ", should_record_loss={should_record_loss}"

        if self.should_apply_projection_back:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(self, **kwargs) -> None:
        r"""Reset attributes following on given keyword arguments.

        Args:
            kwargs:
                Set arguments as attributes of PDSBSSbase.
        """
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        X = self.input

        n_channels, n_bins, n_frames = X.shape
        n_sources = n_channels  # n_channels == n_sources

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        if not hasattr(self, "demix_filter"):
            W = np.eye(n_sources, n_channels, dtype=np.complex128)
            W = np.tile(W, reps=(n_bins, 1, 1))
        else:
            if self.demix_filter is None:
                W = None
            else:
                # To avoid overwriting ``demix_filter`` given by keyword arguments.
                W = self.demix_filter.copy()

        self.demix_filter = W
        self.output = self.separate(X, demix_filter=W)

    def separate(self, input: np.ndarray, demix_filter: np.ndarray) -> np.ndarray:
        r"""Separate ``input`` using ``demixing_filter``.

        .. math::
            \boldsymbol{y}_{ij}
            = \boldsymbol{W}_{i}\boldsymbol{x}_{ij}

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            demix_filter (numpy.ndarray):
                The demixing filters to separate ``input``.
                The shape is (n_bins, n_sources, n_channels).

        Returns:
            numpy.ndarray:
                The separated signal in frequency-domain.
                The shape is (n_sources, n_bins, n_frames).
        """
        X, W = input, demix_filter
        Y = W @ X.transpose(1, 0, 2)
        output = Y.transpose(1, 0, 2)

        return output

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        Returns:
            float:
                Computed loss.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)  # (n_sources, n_bins, n_frames)
        logdet = self.compute_logdet(W)  # (n_bins,)
        penalty = self.penalty_fn(Y)
        loss = penalty - 2 * np.sum(logdet, axis=0)

        return loss

    def compute_logdet(self, demix_filter: np.ndarray) -> np.ndarray:
        r"""Compute log-determinant of demixing filter
        """
        return np.log(np.abs(np.linalg.det(demix_filter)))  # (n_bins,)

    def apply_projection_back(self) -> None:
        r"""Apply projection back technique to estimated spectrograms.
        """
        assert self.should_apply_projection_back, "Set self.should_apply_projection_back=True."

        X, W = self.input, self.demix_filter
        W_scaled = projection_back(W, reference_id=self.reference_id)
        Y_scaled = self.separate(X, demix_filter=W_scaled)

        self.output, self.demix_filter = Y_scaled, W_scaled


class PDSBSS(PDSBSSbase):
    r"""Blind source separation via proximal splitting algorithm [#yatabe2018determined]_.

    Args:
        mu1 (float):
            Step size. Default: ``1``.
        mu2 (float):
            Step size. Default: ``1``.
        alpha (float):
            Step size. Default: ``1``.
        penalty_fn (callable):
            Penalty function that determines source model.
        prox_penalty (callable):
            Proximal operator of penalty function.
            Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_apply_projection_back (bool):
            If ``should_apply_projection_back=True``, the projection back is applied to \
            estimated spectrograms. Default: ``True``.
        should_record_loss (bool):
            Record the loss at each iteration of the update algorithm \
            if ``should_record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.
    """

    def __init__(
        self,
        mu1: float = 1,
        mu2: float = 1,
        alpha: float = 1,
        penalty_fn: Callable[[np.ndarray, np.ndarray], float] = None,
        prox_penalty: Callable[[np.ndarray, float], np.ndarray] = None,
        callbacks: Optional[
            Union[Callable[["PDSBSS"], None], List[Callable[["PDSBSS"], None]]]
        ] = None,
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            penalty_fn=penalty_fn,
            prox_penalty=prox_penalty,
            callbacks=callbacks,
            should_apply_projection_back=should_apply_projection_back,
            should_record_loss=should_record_loss,
            reference_id=reference_id,
        )

        self.mu1, self.mu2 = mu1, mu2
        self.alpha = alpha

    def __call__(self, input, n_iter=100, **kwargs):
        self.input = input.copy()

        self._reset(**kwargs)

        if self.should_record_loss:
            loss = self.compute_loss()
            self.loss.append(loss)

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self)

        for _ in range(n_iter):
            self.update_once()

            if self.should_record_loss:
                loss = self.compute_loss()
                self.loss.append(loss)

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)

        if self.should_apply_projection_back:
            self.apply_projection_back()

        self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "PDSBSS("
        s += "mu1={mu1}, mu2={mu2}"
        s += ", alpha={alpha}"
        s += ", should_apply_projection_back={should_apply_projection_back}"
        s += ", should_record_loss={should_record_loss}"

        if self.should_apply_projection_back:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(self, **kwargs) -> None:
        r"""Reset attributes following on given keyword arguments.

        Args:
            kwargs:
                Set arguments as attributes of PDSBSS.
        """
        super()._reset(**kwargs)

        n_sources = self.n_sources
        n_bins, n_frames = self.n_bins, self.n_frames

        if not hasattr(self, "dual"):
            dual = np.zeros((n_sources, n_bins, n_frames), dtype=np.complex128)
        else:
            if self.dual is None:
                dual = None
            else:
                # To avoid overwriting ``dual`` given by keyword arguments.
                dual = self.dual.copy()

        self.dual = dual

    def update_once(self):
        r"""Update demixing filters and dual parameters once.
        """
        mu1, mu2 = self.mu1, self.mu2
        alpha = self.alpha

        Y = self.dual
        X, W = self.input, self.demix_filter

        XY = np.sum(Y * X[:, np.newaxis, :, :].conj(), axis=-1)
        W_tilde = prox.logdet(W - mu1 * mu2 * XY.transpose(2, 0, 1), step_size=mu1)
        Z = Y + self.separate(X, demix_filter=2 * W_tilde - W)
        Y_tilde = Z - self.prox_penalty(Z, step_size=1 / mu2)

        self.demix_filter = alpha * W_tilde + (1 - alpha) * W
        self.dual = alpha * Y_tilde + (1 - alpha) * Y
