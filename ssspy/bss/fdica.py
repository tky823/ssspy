from typing import Optional, Union, List, Tuple, Callable, Iterable
import functools

import numpy as np

from ._flooring import max_flooring
from ._select_pair import sequential_pair_selector
from ._update_spatial_model import update_by_ip1, update_by_ip2_one_pair
from ._solve_permutation import correlation_based_permutation_solver
from .base import IterativeMethodBase
from ..algorithm import projection_back

__all__ = [
    "GradFDICA",
    "NaturalGradFDICA",
    "AuxFDICA",
    "GradLaplaceFDICA",
    "NaturalGradLaplaceFDICA",
    "AuxLaplaceFDICA",
]

spatial_algorithms = ["IP", "IP1", "IP2"]
EPS = 1e-10


class FDICAbase(IterativeMethodBase):
    r"""Base class of frequency-domain independent component analysis (FDICA).

    Args:
        contrast_fn (callable):
            A contrast function which corresponds to :math:`-\log p(y_{ijn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        solve_permutation (bool):
            If ``solve_permutation=True``, a permutation solver is used to align
            estimated spectrograms. Default: ``True``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back`` explicitly.
            Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back. Default: ``0``.
    """

    def __init__(
        self,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["FDICAbase"], None], List[Callable[["FDICAbase"], None]]]
        ] = None,
        solve_permutation: bool = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(callbacks=callbacks, record_loss=record_loss)

        if contrast_fn is None:
            raise ValueError("Specify contrast function.")
        else:
            self.contrast_fn = contrast_fn

        if flooring_fn is None:
            self.flooring_fn = lambda x: x
        else:
            self.flooring_fn = flooring_fn

        self.input = None
        self.solve_permutation = solve_permutation
        self.scale_restoration = scale_restoration

        if reference_id is None and scale_restoration:
            raise ValueError("Specify 'reference_id' if scale_restoration=True.")
        else:
            self.reference_id = reference_id

    def __call__(self, input: np.ndarray, n_iter: int = 100, **kwargs) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                Mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                Number of iterations of demixing filter updates.
                Default: ``100``.

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_channels, n_bins, n_frames).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        super().__call__(n_iter=n_iter)

        raise NotImplementedError("Implement '__call__' method.")

    def __repr__(self) -> str:
        s = "FDICA("
        s += ", solve_permutation={solve_permutation}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(self, **kwargs) -> None:
        r"""Reset attributes by given keyword arguments.

        Args:
            kwargs:
                Keyword arguments to set as attributes of FDICA.
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
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_sources, n_bins, n_frames).
        """
        X, W = input, demix_filter
        Y = W @ X.transpose(1, 0, 2)
        output = Y.transpose(1, 0, 2)

        return output

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        :math:`\mathcal{L}` is given as follows:

        .. math::
            \mathcal{L}
            &= \sum_{i}\mathcal{L}^{[i]}, \\
            \mathcal{L}^{[i]}
            &= \frac{1}{J}\sum_{j,n}G(y_{ijn})
            - 2\log|\det\boldsymbol{W}_{i}|, \\
            G(y_{ijn}) \
            &= - \log p(y_{ijn})

        Returns:
            Computed loss.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)  # (n_sources, n_bins, n_frames)
        logdet = self.compute_logdet(W)  # (n_bins,)
        G = self.contrast_fn(Y)  # (n_sources, n_bins, n_frames)
        loss = np.sum(np.mean(G, axis=2), axis=0) - 2 * logdet
        loss = loss.sum(axis=0).item()

        return loss

    def compute_logdet(self, demix_filter: np.ndarray) -> np.ndarray:
        r"""Compute log-determinant of demixing filter.

        Args:
            demix_filter (numpy.ndarray):
                Demixing filters with shape of (n_bins, n_sources, n_channels).

        Returns:
            numpy.ndarray of computed log-determinant values.
        """
        _, logdet = np.linalg.slogdet(demix_filter)  # (n_bins,)

        return logdet

    def restore_scale(self) -> None:
        r"""Restore scale ambiguity.

        If ``self.scale_restoration="projection_back``, we use projection back technique.
        """
        scale_restoration = self.scale_restoration

        assert scale_restoration, "Set self.scale_restoration=True."

        if type(scale_restoration) is bool:
            scale_restoration = "projection_back"

        if scale_restoration == "projection_back":
            self.apply_projection_back()
        else:
            raise ValueError("{} is not supported for scale restoration.".format(scale_restoration))

    def apply_projection_back(self) -> None:
        r"""Apply projection back technique to estimated spectrograms."""
        assert self.scale_restoration, "Set self.scale_restoration=True."

        X, W = self.input, self.demix_filter
        W_scaled = projection_back(W, reference_id=self.reference_id)
        Y_scaled = self.separate(X, demix_filter=W_scaled)

        self.output, self.demix_filter = Y_scaled, W_scaled


class GradFDICAbase(FDICAbase):
    r"""Base class of frequency-domain independent component analysis (FDICA) \
    using the gradient descent.

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        contrast_fn (callable):
            A contrast function which corresponds to :math:`-\log p(y_{ijn})`.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        score_fn (callable):
            A score function which corresponds to the partial derivative of the contrast function.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        solve_permutation (bool):
            If ``solve_permutation=True``, a permutation solver is used to align
            estimated spectrograms. Default: ``True``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back`` explicitly.
            Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the gradient descent if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back. Default: ``0``.
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        score_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["GradFDICAbase"], None], List[Callable[["GradFDICAbase"], None]]]
        ] = None,
        solve_permutation: bool = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            contrast_fn=contrast_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            solve_permutation=solve_permutation,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

        self.step_size = step_size

        if score_fn is None:
            raise ValueError("Specify score function.")
        else:
            self.score_fn = score_fn

    def __call__(self, input: np.ndarray, n_iter: int = 100, **kwargs) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain.
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                The number of iterations of demixing filter updates.
                Default: ``100``.

        Returns:
            numpy.ndarray of the separated signal in frequency-domain.
            The shape is (n_channels, n_bins, n_frames).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        # Call __call__ of FDICAbase's parent, i.e. __call__ of IterativeMethod
        super(FDICAbase, self).__call__(n_iter=n_iter)

        if self.solve_permutation:
            Y, W = self.output, self.demix_filter

            self.demix_filter = correlation_based_permutation_solver(
                Y, demix_filter=W, flooring_fn=self.flooring_fn
            )

        if self.scale_restoration:
            self.restore_scale()

        self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "GradFDICA("
        s += "step_size={step_size}"
        s += ", solve_permutation={solve_permutation}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        r"""Update demixing filters once."""
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
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used.
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        solve_permutation (bool):
            If ``solve_permutation=True``, a permutation solver is used to align
            estimated spectrograms. Default: ``True``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back`` explicitly.
            Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the gradient descent if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back. Default: ``0``.

    Examples:
        .. code-block:: python

            def contrast_fn(y):
                return 2 * np.abs(y)

            def score_fn(y):
                denom = np.maximum(np.abs(y), 1e-10)
                return y / denom

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
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["GradFDICA"], None], List[Callable[["GradFDICA"], None]]]
        ] = None,
        is_holonomic: bool = False,
        solve_permutation: bool = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            solve_permutation=solve_permutation,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

        self.is_holonomic = is_holonomic

    def __repr__(self) -> str:
        s = "GradFDICA("
        s += "step_size={step_size}"
        s += ", is_holonomic={is_holonomic}"
        s += ", solve_permutation={solve_permutation}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        r"""Update demixing filters once using the gradient descent.

        If ``is_holonomic=True``, demixing filters are updated as follows:

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\boldsymbol{W}_{i} - \eta\left(\frac{1}{J}\sum_{j}
            \boldsymbol{\phi}(\boldsymbol{y}_{ij})\boldsymbol{y}_{ij}^{\mathsf{H}}
            -\boldsymbol{I}\right)\boldsymbol{W}_{i}^{-\mathsf{H}},

        where

        .. math::
            \boldsymbol{\phi}(\boldsymbol{y}_{ij})
            &= \left(\phi(y_{ij1}),\ldots,\phi(y_{ijn}),\ldots,\phi(y_{ijN})
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
        Y_conj = Y.conj()
        PhiY = np.mean(Phi[:, np.newaxis, :, :] * Y_conj[np.newaxis, :, :, :], axis=-1)
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
            A contrast function corresponds to :math:`-\log p(y_{ijn})`. \
            This function is expected to receive (n_channels, n_bins, n_frames) \
            and return (n_channels, n_bins, n_frames).
        score_fn (callable):
            A score function corresponds to the partial derivative of the contrast function. \
            This function is expected to receive (n_channels, n_bins, n_frames) \
            and return (n_channels, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability. \
            This function is expected to receive (n_channels, n_bins, n_frames) \
            and return (n_channels, n_bins, n_frames). \
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used. \
            Default: ``partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration. \
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used. \
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        solve_permutation (bool):
            If ``solve_permutation=True``, a permutation solver is used to align \
            estimated spectrograms. Default: ``True``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity. \
            If ``scale_restoration=True``, the projection back technique is applied to \
            estimated spectrograms. You can also specify ``"projection_back"`` explicitly. \
            Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the gradient descent \
            if ``record_loss=True``. Default: ``True``.
        reference_id (int):
            Reference channel for projection back. Default: ``0``.

    Examples:
        .. code-block:: python

            def contrast_fn(y):
                return 2 * np.abs(y)

            def score_fn(y):
                denom = np.maximum(np.abs(y), 1e-10)
                return y / denom

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
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["NaturalGradFDICA"], None], List[Callable[["NaturalGradFDICA"], None]]]
        ] = None,
        is_holonomic: bool = False,
        solve_permutation: bool = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            solve_permutation=solve_permutation,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

        self.is_holonomic = is_holonomic

    def __repr__(self) -> str:
        s = "NaturalGradFDICA("
        s += "step_size={step_size}"
        s += ", is_holonomic={is_holonomic}"
        s += ", solve_permutation={solve_permutation}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        r"""Update demixing filters once using the gradient descent.

        If ``is_holonomic=True``, demixing filters are updated as follows:

        .. math::
            \boldsymbol{W}_{i}
            \leftarrow\boldsymbol{W}_{i} - \eta\left(\frac{1}{J}\sum_{j}
            \boldsymbol{\phi}(\boldsymbol{y}_{ij})\boldsymbol{y}_{ij}^{\mathsf{H}}
            -\boldsymbol{I}\right)\boldsymbol{W}_{i},

        where

        .. math::
            \boldsymbol{\phi}(\boldsymbol{y}_{ij})
            &= \left(\phi(y_{ij1}),\ldots,\phi(y_{ijn}),\ldots,\phi(y_{ijN})
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
        Y_conj = Y.conj()
        PhiY = np.mean(Phi[:, np.newaxis, :, :] * Y_conj[np.newaxis, :, :, :], axis=-1)
        PhiY = PhiY.transpose(2, 0, 1)  # (n_bins, n_sources, n_sources)
        eye = np.eye(self.n_sources)

        if self.is_holonomic:
            delta = (PhiY - eye) @ W
        else:
            delta = ((1 - eye) * PhiY) @ W

        W = W - self.step_size * delta

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y


class AuxFDICA(FDICAbase):
    r"""Auxiliary-function-based frequency-domain independent component analysis \
    (AuxFDICA) [#ono2010auxiliary]_.

    Args:
        spatial_algorithm (str):
            Algorithm to update demixing filters. \
            Choose ``"IP"``, ``"IP1"``, or ``"IP2"``. Default: ``"IP"``.
        contrast_fn (callable):
            A contrast function corresponds to :math:`-\log p(y_{ijn})`. \
            This function is expected to receive (n_channels, n_bins, n_frames) \
            and return (n_channels, n_bins, n_frames).
        d_contrast_fn (callable):
            A partial derivative of the real contrast function. \
            This function is expected to receive (n_channels, n_bins, n_frames) \
            and return (n_channels, n_bins, n_frames).
        flooring_fn (callable, optional):
            A flooring function for numerical stability. \
            This function is expected to receive (n_channels, n_bins, n_frames) \
            and return (n_channels, n_bins, n_frames). \
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used. \
            Default: ``partial(max_flooring, eps=1e-10)``.
        pair_selector (callable, optional):
            Selector to choose updaing pair in ``IP2`` and ``ISS2``. \
            If ``None`` is given, ``partial(sequential_pair_selector, sort=True)`` is used. \
            Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration. \
            Default: ``None``.
        solve_permutation (bool):
            If ``solve_permutation=True``, a permutation solver is used to align \
            estimated spectrograms. Default: ``True``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity. \
            If ``scale_restoration=True``, the projection back technique is applied to \
            estimated spectrograms. You can also specify ``"projection_back"`` explicitly. \
            Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the gradient descent \
            if ``record_loss=True``. Default: ``True``.
        reference_id (int):
            Reference channel for projection back. Default: ``0``.

    Examples:
        .. code-block:: python

            def contrast_fn(y):
                return 2 * np.abs(y)

            def d_contrast_fn(y):
                return 2 * np.ones_like(y)

            n_channels, n_bins, n_frames = 2, 2049, 128
            spectrogram_mix = np.random.randn(n_channels, n_bins, n_frames) \
                + 1j * np.random.randn(n_channels, n_bins, n_frames)

            fdica = AuxFDICA(contrast_fn=contrast_fn, d_contrast_fn=d_contrast_fn)
            spectrogram_est = fdica(spectrogram_mix, n_iter=100)
            print(spectrogram_mix.shape, spectrogram_est.shape)
            >>> (2, 2049, 128), (2, 2049, 128)

    .. [#ono2010auxiliary]
        Ono, Nobutaka and Miyabe, Shigeki,
        "Auxiliary-function-based independent component analysis for super-Gaussian sources,"
        in *Proc. LVA/ICA*, 2010, pp.165-172.
    """

    def __init__(
        self,
        spatial_algorithm: str = "IP",
        contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        d_contrast_fn: Callable[[np.ndarray], np.ndarray] = None,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        pair_selector: Optional[Callable[[int], Iterable[Tuple[int, int]]]] = None,
        callbacks: Optional[
            Union[Callable[["AuxFDICA"], None], List[Callable[["AuxFDICA"], None]]]
        ] = None,
        solve_permutation: bool = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        super().__init__(
            contrast_fn=contrast_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            solve_permutation=solve_permutation,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )
        assert spatial_algorithm in spatial_algorithms, "Not support {}.".format(spatial_algorithms)

        self.spatial_algorithm = spatial_algorithm
        self.d_contrast_fn = d_contrast_fn

        if pair_selector is None:
            if spatial_algorithm == "IP2":
                self.pair_selector = sequential_pair_selector
        else:
            self.pair_selector = pair_selector

    def __call__(self, input: np.ndarray, n_iter: int = 100, **kwargs) -> np.ndarray:
        r"""Separate a frequency-domain multichannel signal.

        Args:
            input (numpy.ndarray):
                The mixture signal in frequency-domain. \
                The shape is (n_channels, n_bins, n_frames).
            n_iter (int):
                The number of iterations of demixing filter updates.
                Default: ``100``.

        Returns:
            numpy.ndarray:
                The separated signal in frequency-domain. \
                The shape is (n_channels, n_bins, n_frames).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        # Call __call__ of FDICAbase's parent, i.e. __call__ of IterativeMethod
        super(FDICAbase, self).__call__(n_iter=n_iter)

        if self.solve_permutation:
            Y, W = self.output, self.demix_filter

            self.demix_filter = correlation_based_permutation_solver(
                Y, demix_filter=W, flooring_fn=self.flooring_fn
            )

        if self.scale_restoration:
            self.restore_scale()

        self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "AuxFDICA("
        s += "spatial_algorithm={spatial_algorithm}"
        s += ", solve_permutation={solve_permutation}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        r"""Update demixing filters once.

        - If ``self.spatial_algorithm`` is ``"IP"`` or ``"IP1"``, ``update_once_ip1`` is called.
        - If ``self.spatial_algorithm`` is ``"IP2"``, ``update_once_ip2`` is called.
        """
        if self.spatial_algorithm in ["IP", "IP1"]:
            self.update_once_ip1()
        elif self.spatial_algorithm in ["IP2"]:
            self.update_once_ip2()
        else:
            raise NotImplementedError("Not support {}.".format(self.spatial_algorithm))

    def update_once_ip1(self) -> None:
        r"""Update demixing filters once using iterative projection.

        Demixing filters are updated sequentially for :math:`n=1,\ldots,N` as follows:

        .. math::
            \boldsymbol{w}_{in}
            &\leftarrow\left(\boldsymbol{W}_{in}^{\mathsf{H}}\boldsymbol{U}_{in}\right)^{-1}
            \boldsymbol{e}_{n}, \\
            \boldsymbol{w}_{in}
            &\leftarrow\frac{\boldsymbol{w}_{in}}
            {\sqrt{\boldsymbol{w}_{in}^{\mathsf{H}}\boldsymbol{U}_{in}\boldsymbol{w}_{in}}}, \\

        where

        .. math::
            \boldsymbol{U}_{in}
            &= \frac{1}{J}\sum_{j}
            \frac{G'_{\mathbb{R}}(|y_{ijn}|)}{2|y_{ijn}|}
            \boldsymbol{x}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}, \\
            G(y_{ijn})
            &= -\log p(y_{ijn}), \\
            G_{\mathbb{R}}(|y_{ijn}|)
            &= G(y_{ijn}).
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        XX_Hermite = X[:, np.newaxis, :, :] * X[np.newaxis, :, :, :].conj()
        XX_Hermite = XX_Hermite.transpose(2, 0, 1, 3)  # (n_bins, n_channels, n_channels, n_frames)
        Y_abs = np.abs(Y)
        denom = self.flooring_fn(2 * Y_abs)
        varphi = self.d_contrast_fn(Y_abs) / denom  # (n_sources, n_bins, n_frames)
        varphi = varphi.transpose(1, 0, 2)  # (n_bins, n_sources, n_frames)
        GXX = varphi[:, :, np.newaxis, np.newaxis, :] * XX_Hermite[:, np.newaxis, :, :, :]
        U = np.mean(GXX, axis=-1)  # (n_bins, n_sources, n_channels, n_channels)

        self.demix_filter = update_by_ip1(W, U, flooring_fn=self.flooring_fn)

    def update_once_ip2(self) -> None:
        r"""Update demixing filters once using pairwise iterative projection.

        For :math:`n_{1}` and :math:`n_{2}` (:math:`n_{1}\neq n_{2}`), \
        compute auxiliary variables:

        .. math::
            \bar{r}_{ijn_{1}}
            &\leftarrow|y_{ijn_{1}}| \\
            \bar{r}_{ijn_{2}}
            &\leftarrow|y_{ijn_{2}}|

        Then, compute weighted covariance matrix as follows:

        .. math::
            \boldsymbol{G}_{in_{1}}^{(n_{1},n_{2})}
            &= \frac{1}{J}\sum_{j}\frac{G'_{\mathbb{R}}(\bar{r}_{ijn_{1}})}
            {2\bar{r}_{ijn_{1}}} \
            \boldsymbol{y}_{ij}^{(n_{1},n_{2})}{\boldsymbol{y}_{ij}^{(n_{1},n_{2})}}^{\mathsf{H}} \\
            \boldsymbol{G}_{in_{2}}^{(n_{1},n_{2})}
            &= \frac{1}{J}\sum_{j}\frac{G'_{\mathbb{R}}(\bar{r}_{ijn_{2}})}
            {2\bar{r}_{ijn_{2}}} \
            \boldsymbol{y}_{ij}^{(n_{1},n_{2})}{\boldsymbol{y}_{ij}^{(n_{1},n_{2})}}^{\mathsf{H}},

        where

        .. math::
            G(y_{ijn})
            &= -\log p(y_{ijn}), \\
            G_{\mathbb{R}}(|y_{ijn}|)
            &= G(y_{ijn}) \\
            \boldsymbol{y}_{ij}^{(n_{1},n_{2})}
            &= \left(
            \begin{array}{c}
                \boldsymbol{w}_{in_{1}}^{\mathsf{H}}\boldsymbol{x}_{ij} \\
                \boldsymbol{w}_{in_{2}}^{\mathsf{H}}\boldsymbol{x}_{ij}
            \end{array}
            \right).

        Compute generalized eigenvectors of :math:`\boldsymbol{G}_{in_{1}}^{(n_{1},n_{2})}` \
        and :math:`\boldsymbol{G}_{in_{2}}^{(n_{1},n_{2})}`.

        .. math::
            \boldsymbol{G}_{in_{1}}^{(n_{1},n_{2})}\boldsymbol{h}_{i}
            = \lambda_{i}^{(n_{1},n_{2})}\boldsymbol{G}_{in_{2}}^{(n_{1},n_{2})}\boldsymbol{h}_{i}.

        We denote two eigenvectors as :math:`\boldsymbol{h}_{in_{1}}` \
        and :math:`\boldsymbol{h}_{in_{2}}`.

        .. math::
            \boldsymbol{h}_{in_{1}}
            &\leftarrow\frac{\boldsymbol{h}_{in_{1}}}
            {\sqrt{\boldsymbol{h}_{in_{1}}^{\mathsf{H}}\boldsymbol{G}_{in_{1}}^{(n_{1},n_{2})}
            \boldsymbol{h}_{in_{1}}}}, \\
            \boldsymbol{h}_{in_{2}}
            &\leftarrow\frac{\boldsymbol{h}_{in_{2}}}
            {\sqrt{\boldsymbol{h}_{in_{2}}^{\mathsf{H}}\boldsymbol{G}_{in_{2}}^{(n_{1},n_{2})}
            \boldsymbol{h}_{in_{2}}}}.

        Then, update :math:`\boldsymbol{w}_{in_{1}}` and :math:`\boldsymbol{w}_{in_{2}}`
        simultaneously.

        .. math::
            (
            \begin{array}{cc}
                \boldsymbol{w}_{in_{1}} & \boldsymbol{w}_{in_{2}}
            \end{array}
            )\leftarrow(
            \begin{array}{cc}
                \boldsymbol{w}_{in_{1}} & \boldsymbol{w}_{in_{2}}
            \end{array}
            )(
            \begin{array}{cc}
                \boldsymbol{h}_{in_{1}} & \boldsymbol{h}_{in_{2}}
            \end{array}
            )

        At each iteration, we update for all pairs of :math:`n_{1}` \
        and :math:`n_{1}` (:math:`n_{1}<n_{2}`).
        """
        n_sources = self.n_sources

        X, W = self.input, self.demix_filter

        for m, n in self.pair_selector(n_sources):
            W_mn = W[:, (m, n), :]
            Y_mn = self.separate(X, demix_filter=W_mn)

            Y_mn_abs = np.abs(Y_mn)
            denom_mn = self.flooring_fn(2 * Y_mn_abs)
            weight_mn = self.d_contrast_fn(Y_mn_abs) / denom_mn

            W[:, (m, n), :] = update_by_ip2_one_pair(
                Y_mn,
                demix_filter_pair=W_mn,
                weight_pair=weight_mn,
                flooring_fn=self.flooring_fn,
            )

        self.demix_filter = W


class GradLaplaceFDICA(GradFDICA):
    r"""Frequency-domain independent component analysis (FDICA) \
    using the gradient descent on a Laplace distribution.

    We assume :math:`y_{ijn}` follows a Laplace distribution.

    .. math::
        p(y_{ijn})\propto\exp(|y_{ijn}|)

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability. \
            This function is expected to receive (n_channels, n_bins, n_frames)
            and return (n_channels, n_bins, n_frames).
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration. \
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used. \
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        solve_permutation (bool):
            If ``solve_permutation=True``, a permutation solver is used to align \
            estimated spectrograms. Default: ``True``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity. \
            If ``scale_restoration=True``, the projection back technique is applied to \
            estimated spectrograms. You can also specify ``"projection_back"`` explicitly. \
            Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the gradient descent \
            if ``record_loss=True``. Default: ``True``.
        reference_id (int):
            Reference channel for projection back. Default: ``0``.

    Examples:
        .. code-block:: python

            n_channels, n_bins, n_frames = 2, 2049, 128
            spectrogram_mix = \
                np.random.randn(n_channels, n_bins, n_frames) \
                + 1j * np.random.randn(n_channels, n_bins, n_frames)

            fdica = GradLaplaceFDICA()
            spectrogram_est = fdica(spectrogram_mix, n_iter=1000)
            print(spectrogram_mix.shape, spectrogram_est.shape)
            >>> (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["GradLaplaceFDICA"], None], List[Callable[["GradLaplaceFDICA"], None]]]
        ] = None,
        is_holonomic: bool = False,
        solve_permutation: bool = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        def contrast_fn(y: np.ndarray) -> np.ndarray:
            r"""Contrast function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray:
                    The shape is (n_sources, n_bins, n_frames).
            """
            return 2 * np.abs(y)

        def score_fn(y: np.ndarray) -> np.ndarray:
            r"""Score function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray:
                    The shape is (n_sources, n_bins, n_frames).
            """
            denom = self.flooring_fn(np.abs(y))
            return y / denom

        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            is_holonomic=is_holonomic,
            solve_permutation=solve_permutation,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

    def __repr__(self) -> str:
        s = "GradLaplaceFDICA("
        s += "step_size={step_size}"
        s += ", is_holonomic={is_holonomic}"
        s += ", solve_permutation={solve_permutation}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)


class NaturalGradLaplaceFDICA(GradFDICA):
    r"""Frequency-domain independent component analysis (FDICA) \
    using the natural gradient descent on a Laplace distribution.

    We assume :math:`y_{ijn}` follows a Laplace distribution.

    .. math::
        p(y_{ijn})\propto\exp(|y_{ijn}|)

    Args:
        step_size (float):
            A step size of the gradient descent. Default: ``1e-1``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability. \
            This function is expected to receive (n_channels, n_bins, n_frames) \
            and return (n_channels, n_bins, n_frames). \
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used.
            Default: ``partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration. \
            Default: ``None``.
        is_holonomic (bool):
            If ``is_holonomic=True``, Holonomic-type update is used. \
            Otherwise, Nonholonomic-type update is used. Default: ``False``.
        solve_permutation (bool):
            If ``solve_permutation=True``, a permutation solver is used to align \
            estimated spectrograms. Default: ``True``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity. \
            If ``scale_restoration=True``, the projection back technique is applied to \
            estimated spectrograms. You can also specify ``"projection_back"`` explicitly. \
            Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the gradient descent \
            if ``record_loss=True``. Default: ``True``.
        reference_id (int):
            Reference channel for projection back. Default: ``0``.

    Examples:
        .. code-block:: python

            n_channels, n_bins, n_frames = 2, 2049, 128
            spectrogram_mix = \
                np.random.randn(n_channels, n_bins, n_frames) \
                + 1j * np.random.randn(n_channels, n_bins, n_frames)

            fdica = NaturalGradLaplaceFDICA()
            spectrogram_est = fdica(spectrogram_mix, n_iter=1000)
            print(spectrogram_mix.shape, spectrogram_est.shape)
            >>> (2, 2049, 128), (2, 2049, 128)
    """

    def __init__(
        self,
        step_size: float = 1e-1,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[
                Callable[["NaturalGradLaplaceFDICA"], None],
                List[Callable[["NaturalGradLaplaceFDICA"], None]],
            ]
        ] = None,
        is_holonomic: bool = False,
        solve_permutation: bool = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        def contrast_fn(y: np.ndarray) -> np.ndarray:
            r"""Contrast function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray:
                    The shape is (n_sources, n_bins, n_frames).
            """
            return 2 * np.abs(y)

        def score_fn(y: np.ndarray) -> np.ndarray:
            r"""Score function.

            Args:
                y (numpy.ndarray):
                    The shape is (n_sources, n_bins, n_frames).

            Returns:
                numpy.ndarray:
                    The shape is (n_sources, n_bins, n_frames).
            """
            denom = self.flooring_fn(np.abs(y))
            return y / denom

        super().__init__(
            step_size=step_size,
            contrast_fn=contrast_fn,
            score_fn=score_fn,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            is_holonomic=is_holonomic,
            solve_permutation=solve_permutation,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

    def __repr__(self) -> str:
        s = "NaturalGradLaplaceFDICA("
        s += "step_size={step_size}"
        s += ", is_holonomic={is_holonomic}"
        s += ", solve_permutation={solve_permutation}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)


class AuxLaplaceFDICA(AuxFDICA):
    r"""Auxiliary-function-based frequency-domain independent component analysis \
    on a Laplace distribution.

    We assume :math:`y_{ijn}` follows a Laplace distribution.

    .. math::
        p(y_{ijn})\propto\exp(|y_{ijn}|)

    Args:
        spatial_algorithm (str):
            Algorithm to update demixing filters. \
            Choose ``"IP"``, ``"IP1"``, or ``"IP2"``. Default: ``"IP"``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability. \
            This function is expected to receive (n_channels, n_bins, n_frames) \
            and return (n_channels, n_bins, n_frames).
            If you explicitly set ``flooring_fn=None``, \
            the identity function (``lambda x: x``) is used. \
            Default: ``partial(max_flooring, eps=1e-10)``.
        pair_selector (callable, optional):
            Selector to choose updaing pair in ``IP2`` and ``ISS2``. \
            If ``None`` is given, ``partial(sequential_pair_selector, sort=True)`` is used. \
            Default: ``None``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration. \
            Default: ``None``.
        solve_permutation (bool):
            If ``solve_permutation=True``, a permutation solver is used to align \
            estimated spectrograms. Default: ``True``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity. \
            If ``scale_restoration=True``, the projection back technique is applied to \
            estimated spectrograms. You can also specify ``"projection_back"`` explicitly. \
            Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the gradient descent \
            if ``record_loss=True``. Default: ``True``.
        reference_id (int):
            Reference channel for projection back. Default: ``0``.
    """

    def __init__(
        self,
        spatial_algorithm: str = "IP",
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        pair_selector: Optional[Callable[[int], Iterable[Tuple[int, int]]]] = None,
        callbacks: Optional[
            Union[Callable[["AuxLaplaceFDICA"], None], List[Callable[["AuxLaplaceFDICA"], None]]]
        ] = None,
        solve_permutation: bool = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
    ) -> None:
        def contrast_fn(y: np.ndarray):
            return 2 * np.abs(y)

        def d_contrast_fn(y: np.ndarray):
            return 2 * np.ones_like(y)

        super().__init__(
            spatial_algorithm=spatial_algorithm,
            contrast_fn=contrast_fn,
            d_contrast_fn=d_contrast_fn,
            flooring_fn=flooring_fn,
            pair_selector=pair_selector,
            callbacks=callbacks,
            solve_permutation=solve_permutation,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
        )

    def __repr__(self) -> str:
        s = "AuxLaplaceFDICA("
        s += "spatial_algorithm={spatial_algorithm}"
        s += ", solve_permutation={solve_permutation}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)
