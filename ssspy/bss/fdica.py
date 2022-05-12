from typing import Optional, Union, List, Callable
import itertools
from functools import partial

import numpy as np

from ._flooring import max_flooring
from ..algorithm import projection_back

__all__ = ["GradFDICA", "NaturalGradFDICA"]

EPS = 1e-12


class FDICAbase:
    r"""Base class of frequency-domain independent component analysis (FDICA).

    Args:
        contrast_fn (callable):
            A contrast function corresponds to :math:`-\log p(y_{ijn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        score_fn (callable):
            A score function corresponds to the partial derivative of the contrast function.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``partial(max_flooring, eps=1e-12)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_solve_permutation (bool):
            If ``should_solve_permutation=True``, a permutation solver is used to align \
            estimated spectrograms. Default: ``True``.
        should_apply_projection_back (bool):
            If ``should_apply_projection_back=True``, the projection back is applied to \
            estimated spectrograms. Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.
        should_record_loss (bool):
            Record the loss at each iteration of the update algorithm \
            if ``should_record_loss=True``.
            Default: ``True``.
    """

    def __init__(
        self,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        score_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = partial(max_flooring, eps=EPS),
        callbacks: Optional[
            Union[Callable[["FDICAbase"], None], List[Callable[["FDICAbase"], None]]]
        ] = None,
        should_solve_permutation: bool = True,
        should_apply_projection_back: bool = True,
        reference_id: int = 0,
        should_record_loss: bool = True,
    ) -> None:
        if contrast_fn is None:
            raise ValueError("Specify contrast function.")
        else:
            self.contrast_fn = contrast_fn

        if score_fn is None:
            raise ValueError("Specify score function.")
        else:
            self.score_fn = score_fn

        if flooring_fn is None:
            self.flooring_fn = lambda x: x
        else:
            self.flooring_fn = flooring_fn

        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None

        self.input = None
        self.should_solve_permutation = should_solve_permutation
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
        r"""Separate a time-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                Mixture signal in time-domain.
                The shape is (n_channels, n_samples).
            n_iter (int):
                Number of iterations of demixing filter updates.
                Default: 100.

        Returns:
            numpy.ndarray:
                The separated signal in time-domain.
                The shape is (n_sources, n_samples).
        """
        raise NotImplementedError("Implement '__call__' method.")

    def __repr__(self) -> str:
        s = "FDICA("
        s += ", should_solve_permutation={should_solve_permutation}"
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
                Set arguments as attributes of FDICA.
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

    def compute_negative_loglikelihood(self) -> float:
        r"""Compute negative log-likelihood :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L} \
            &= \sum_{i}\mathcal{L}^{[i]}, \\
            \mathcal{L}^{[i]} \
            &= \frac{1}{J}\sum_{j,n}G(y_{ijn}) \
            - 2\log|\det\boldsymbol{W}_{i}|, \\
            G(y_{ijn}) \
            &= - \log p(y_{ijn})

        Returns:
            float:
                Computed negative log-likelihood.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)  # (n_channels, n_bins, n_frames)
        logdet = np.log(np.abs(np.linalg.det(W)))  # (n_bins,)
        G = self.contrast_fn(Y)  # (n_channels, n_bins, n_frames)
        loss = np.sum(np.mean(G, axis=2), axis=0) - 2 * logdet
        loss = loss.sum(axis=0)

        return loss

    def solve_permutation(self) -> None:
        r"""Solve permutaion of estimated spectrograms.

        Group channels at each frequency bin according to correlations
        between frequencies [#sawada2010underdetermined]_.

        .. [#sawada2010underdetermined]
            H. Sawada, S. Araki, and S. Makino,
            "Underdetermined convolutive blind source separation \
            via frequency bin-wise clustering and permutation alignment,"
            in *IEEE Trans. ASLP*, vol. 19, no. 3, pp. 516-527, 2010.
        """
        assert self.should_solve_permutation, "Set self.should_solve_permutation=True."

        n_sources, n_bins = self.n_sources, self.n_bins
        Y, W = self.output, self.demix_filter

        permutations = list(itertools.permutations(range(n_sources)))

        P = np.abs(Y).transpose(1, 0, 2)  # (n_bins, n_sources, n_frames)
        norm = np.sqrt(np.sum(P ** 2, axis=1, keepdims=True))
        norm = self.flooring_fn(norm)
        P = P / norm
        correlation = np.sum(P @ P.transpose(0, 2, 1), axis=(1, 2))
        indices = np.argsort(correlation)

        min_idx = indices[0]
        P_criteria = P[min_idx]

        for bin_idx in range(1, n_bins):
            min_idx = indices[bin_idx]
            P_max = None
            perm_max = None

            for perm in permutations:
                P_perm = np.sum(P_criteria * P[min_idx, perm, :])

                if P_max is None or P_perm > P_max:
                    P_max = P_perm
                    perm_max = perm

            P_criteria = P_criteria + P[min_idx, perm_max, :]
            W[min_idx, :, :] = W[min_idx, perm_max, :]

        self.demix_filter = W

    def apply_projection_back(self) -> None:
        r"""Apply projection back technique to estimated spectrograms.
        """
        assert self.should_apply_projection_back, "Set self.should_apply_projection_back=True."

        X, W = self.input, self.demix_filter
        W_scaled = projection_back(W, reference_id=self.reference_id)
        output = self.separate(X, demix_filter=W_scaled)

        self.output = output


class GradFDICAbase(FDICAbase):
    r"""Base class of frequency-domain independent component analysis (FDICA) \
    using the gradient descent.

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        contrast_fn (callable):
            A contrast function corresponds to :math:`-\log p(y_{ijn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        score_fn (callable):
            A score function corresponds to the partial derivative of the contrast function.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``partial(max_flooring, eps=1e-12)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        should_solve_permutation (bool):
            If ``should_solve_permutation=True``, a permutation solver is used to align \
            estimated spectrograms. Default: ``True``.
        should_apply_projection_back (bool):
            If ``should_apply_projection_back=True``, the projection back is applied to \
            estimated spectrograms. Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.
        should_record_loss (bool):
            Record the loss at each iteration of the gradient descent \
            if ``should_record_loss=True``.
            Default: ``True``.
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        score_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = partial(max_flooring, eps=EPS),
        callbacks: Optional[
            Union[Callable[["GradFDICAbase"], None], List[Callable[["GradFDICAbase"], None]]]
        ] = None,
        should_solve_permutation: bool = True,
        should_apply_projection_back: bool = True,
        reference_id: int = 0,
        should_record_loss: bool = True,
    ) -> None:
        super().__init__(
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            should_solve_permutation=should_solve_permutation,
            should_apply_projection_back=should_apply_projection_back,
            reference_id=reference_id,
            should_record_loss=should_record_loss,
        )

        self.step_size = step_size

    def __call__(self, input: np.ndarray, n_iter: int = 100, **kwargs) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                The number of iterations of demixing filter updates.
                Default: 100.

        Returns:
            numpy.ndarray:
                The separated signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        if self.should_record_loss:
            loss = self.compute_negative_loglikelihood()
            self.loss.append(loss)

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self)

        for _ in range(n_iter):
            self.update_once()

            if self.should_record_loss:
                loss = self.compute_negative_loglikelihood()
                self.loss.append(loss)

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)

        if self.should_solve_permutation:
            self.solve_permutation()

        if self.should_apply_projection_back:
            self.apply_projection_back()

        self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "GradFDICA("
        s += "step_size={step_size}"
        s += ", should_solve_permutation={should_solve_permutation}"
        s += ", should_apply_projection_back={should_apply_projection_back}"
        s += ", should_record_loss={should_record_loss}"

        if self.should_apply_projection_back:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        r"""Update demixing filters once.
        """
        raise NotImplementedError("Implement 'update_once' method.")


class GradFDICA(GradFDICAbase):
    r"""Frequency-domain independent component analysis (FDICA) \
    using the gradient descent.

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        contrast_fn (callable):
            A contrast function corresponds to :math:`-\log p(y_{ijn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        score_fn (callable):
            A score function corresponds to the partial derivative of the contrast function.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``partial(max_flooring, eps=1e-12)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        should_solve_permutation (bool):
            If ``should_solve_permutation=True``, a permutation solver is used to align \
            estimated spectrograms. Default: ``True``.
        should_apply_projection_back (bool):
            If ``should_apply_projection_back=True``, the projection back is applied to \
            estimated spectrograms. Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.
        should_record_loss (bool):
            Record the loss at each iteration of the gradient descent \
            if ``should_record_loss=True``.
            Default: ``True``.

    Examples:
        .. code-block:: python

            def contrast_fn(y):
                return 2 * np.abs(y)

            def score_fn(y):
                denominator = np.maximum(np.abs(y), 1e-12)
                return y / denominator

            n_channels, n_bins, n_frames = 2, 2049, 128
            spectrogram_mix = \
                np.random.randn(n_channels, n_bins, n_frames) \
                + 1j * np.random.randn(n_channels, n_bins, n_frames)

            fdica = GradFDICA(contrast_fn=contrast_fn, score_fn=score_fn)
            spectrogram_est = fdica(spectrogram_mix, n_iter=1000)
            print(spectrogram_mix.shape, spectrogram_est.shape)
            >>> (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        score_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = partial(max_flooring, eps=EPS),
        callbacks: Optional[
            Union[Callable[["GradFDICA"], None], List[Callable[["GradFDICA"], None]]]
        ] = None,
        is_holonomic: bool = False,
        should_solve_permutation: bool = True,
        should_apply_projection_back: bool = True,
        reference_id: int = 0,
        should_record_loss: bool = True,
    ) -> None:
        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            should_solve_permutation=should_solve_permutation,
            should_apply_projection_back=should_apply_projection_back,
            reference_id=reference_id,
            should_record_loss=should_record_loss,
        )

        self.is_holonomic = is_holonomic

    def __repr__(self) -> str:
        s = "GradFDICA("
        s += "step_size={step_size}"
        s += ", is_holonomic={is_holonomic}"
        s += ", should_solve_permutation={should_solve_permutation}"
        s += ", should_apply_projection_back={should_apply_projection_back}"
        s += ", should_record_loss={should_record_loss}"

        if self.should_apply_projection_back:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        r"""Update demixing filters once using the gradient descent.

        If ``is_holonomic=True``, demixing filters are updated as follows:

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\boldsymbol{W}_{i} - \eta\left(\frac{1}{J}\sum_{j} \
            \boldsymbol{\phi}(\boldsymbol{y}_{ij})\boldsymbol{y}_{ij}^{\mathsf{H}} \
            -\boldsymbol{I}\right)\boldsymbol{W}_{i}^{-\mathsf{H}},

        where

        .. math::
            \boldsymbol{\phi}(\boldsymbol{y}_{ij})
            &= \left(\phi(y_{ij1}),\ldots,\phi(y_{ijn}),\ldots,\phi(y_{ijN})\
            \right)^{\mathsf{T}}\in\mathbb{C}^{N}, \\
            \phi(y_{ijn})
            &= \frac{\partial G(y_{ijn})}{\partial y_{ijn}^{*}}, \\
            G(y_{ijn})
            &= -\log p(y_{ijn}).

        Otherwise (``is_holonomic=False``),

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\boldsymbol{W}_{i}
            - \eta\cdot\mathrm{offdiag}\left(\frac{1}{J}\sum_{j}
            \boldsymbol{\phi}(\boldsymbol{y}_{ij})\boldsymbol{y}_{ij}^{\mathsf{H}}\right)
            \boldsymbol{W}_{i}^{-\mathsf{H}}.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        Phi = self.score_fn(Y)
        PhiY = np.mean(Phi[:, np.newaxis, :, :] * Y[np.newaxis, :, :, :], axis=-1)
        PhiY = PhiY.transpose(2, 0, 1)  # (n_bins, n_sources, n_sources)
        W_inv = np.linalg.inv(W)
        W_inv_Hermite = W_inv.transpose(0, 2, 1).conj()
        eye = np.eye(self.n_sources)

        if self.is_holonomic:
            delta = (PhiY - eye) @ W_inv_Hermite
        else:
            delta = ((1 - eye) * PhiY) @ W_inv_Hermite

        W = W - self.step_size * delta

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y


class NaturalGradFDICA(GradFDICAbase):
    r"""Frequency-domain independent component analysis (FDICA) \
    using the natural gradient descent.

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        contrast_fn (callable):
            A contrast function corresponds to :math:`-\log p(y_{ijn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        score_fn (callable):
            A score function corresponds to the partial derivative of the contrast function.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``partial(max_flooring, eps=1e-12)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        should_solve_permutation (bool):
            If ``should_solve_permutation=True``, a permutation solver is used to align \
            estimated spectrograms. Default: ``True``.
        should_apply_projection_back (bool):
            If ``should_apply_projection_back=True``, the projection back is applied to \
            estimated spectrograms. Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.
        should_record_loss (bool):
            Record the loss at each iteration of the gradient descent \
            if ``should_record_loss=True``.
            Default: ``True``.

    Examples:
        .. code-block:: python

            def contrast_fn(y):
                return 2 * np.abs(y)

            def score_fn(y):
                denominator = np.maximum(np.abs(y), 1e-12)
                return y / denominator

            n_channels, n_bins, n_frames = 2, 2049, 128
            spectrogram_mix = \
                np.random.randn(n_channels, n_bins, n_frames) \
                + 1j * np.random.randn(n_channels, n_bins, n_frames)

            fdica = NaturalGradFDICA(contrast_fn=contrast_fn, score_fn=score_fn)
            spectrogram_est = fdica(spectrogram_mix, n_iter=1000)
            print(spectrogram_mix.shape, spectrogram_est.shape)
            >>> (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        score_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = partial(max_flooring, eps=EPS),
        callbacks: Optional[
            Union[Callable[["GradFDICA"], None], List[Callable[["GradFDICA"], None]]]
        ] = None,
        is_holonomic: bool = False,
        should_solve_permutation: bool = True,
        should_apply_projection_back: bool = True,
        reference_id: int = 0,
        should_record_loss: bool = True,
    ) -> None:
        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            should_solve_permutation=should_solve_permutation,
            should_apply_projection_back=should_apply_projection_back,
            reference_id=reference_id,
            should_record_loss=should_record_loss,
        )

        self.is_holonomic = is_holonomic

    def __repr__(self) -> str:
        s = "NaturalGradFDICA("
        s += "step_size={step_size}"
        s += ", is_holonomic={is_holonomic}"
        s += ", should_solve_permutation={should_solve_permutation}"
        s += ", should_apply_projection_back={should_apply_projection_back}"
        s += ", should_record_loss={should_record_loss}"

        if self.should_apply_projection_back:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        r"""Update demixing filters once using the gradient descent.

        If ``is_holonomic=True``, demixing filters are updated as follows:

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\boldsymbol{W}_{i} - \eta\left(\frac{1}{J}\sum_{j} \
            \boldsymbol{\phi}(\boldsymbol{y}_{ij})\boldsymbol{y}_{ij}^{\mathsf{H}} \
            -\boldsymbol{I}\right)\boldsymbol{W}_{i},

        where

        .. math::
            \boldsymbol{\phi}(\boldsymbol{y}_{ij})
            &= \left(\phi(y_{ij1}),\ldots,\phi(y_{ijn}),\ldots,\phi(y_{ijN})\
            \right)^{\mathsf{T}}\in\mathbb{C}^{N}, \\
            \phi(y_{ijn})
            &= \frac{\partial G(y_{ijn})}{\partial y_{ijn}^{*}}, \\
            G(y_{ijn})
            &= -\log p(y_{ijn}).

        Otherwise (``is_holonomic=False``),

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\boldsymbol{W}_{i}
            - \eta\cdot\mathrm{offdiag}\left(\frac{1}{J}\sum_{j}
            \boldsymbol{\phi}(\boldsymbol{y}_{ij})\boldsymbol{y}_{ij}^{\mathsf{H}}\right)
            \boldsymbol{W}_{i}.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        Phi = self.score_fn(Y)
        PhiY = np.mean(Phi[:, np.newaxis, :, :] * Y[np.newaxis, :, :, :], axis=-1)
        PhiY = PhiY.transpose(2, 0, 1)  # (n_bins, n_sources, n_sources)
        W_inv = np.linalg.inv(W)
        W_inv_Hermite = W_inv.transpose(0, 2, 1).conj()
        eye = np.eye(self.n_sources)

        if self.is_holonomic:
            delta = (PhiY - eye) @ W_inv_Hermite
        else:
            delta = ((1 - eye) * PhiY) @ W_inv_Hermite

        W = W - self.step_size * delta

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y
