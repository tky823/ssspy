import functools
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from ..algorithm import (
    MINIMAL_DISTORTION_PRINCIPLE_KEYWORDS,
    PROJECTION_BACK_KEYWORDS,
    minimal_distortion_principle,
    projection_back,
)
from ..linalg.mean import gmeanmh
from ..linalg.quadratic import quadratic
from ..linalg.sqrtm import invsqrtmh, sqrtmh
from ..special.flooring import identity, max_flooring
from ..special.psd import to_psd
from ..utils.flooring import choose_flooring_fn
from ._update_spatial_model import update_by_block_decomposition_vcd
from .base import IterativeMethodBase

spatial_algorithms = ["FPI", "VCD"]
source_algorithms = ["EM", "MM"]
EPS = 1e-10


class IPSDTABase(IterativeMethodBase):
    r"""Base class of independent positive semidefinite tensor analysis (IPSDTA).

    Args:
        n_basis (int):
            Number of PSDTF bases.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back`` explicitly.
            Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.
        rng (numpy.random.Generator, optioinal):
            Random number generator. This is mainly used to randomly initialize PSDTF.
            If ``None`` is given, ``np.random.default_rng()`` is used.
            Default: ``None``.
    """

    def __init__(
        self,
        n_basis: int,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[Callable[["IPSDTABase"], None], List[Callable[["IPSDTABase"], None]]]
        ] = None,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.source_normalization: Optional[Union[bool, str]]

        super().__init__(callbacks=callbacks, record_loss=record_loss)

        self.n_basis = n_basis

        if flooring_fn is None:
            self.flooring_fn = identity
        else:
            self.flooring_fn = flooring_fn

        self.input = None
        self.scale_restoration = scale_restoration

        if reference_id is None and scale_restoration:
            raise ValueError("Specify 'reference_id' if scale_restoration=True.")
        else:
            self.reference_id = reference_id

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

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

        super().__call__(n_iter=n_iter)

        if self.scale_restoration:
            self.restore_scale()

        self.output = self.separate(self.input, demix_filter=self.demix_filter)

        return self.output

    def __repr__(self) -> str:
        s = "IPSDTA("
        s += "n_basis={n_basis}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
        **kwargs,
    ) -> None:
        r"""Reset attributes by given keyword arguments.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.
            kwargs:
                Keyword arguments to set as attributes of IPSDTA.
        """
        assert self.input is not None, "Specify data!"

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

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

        self._init_psdtf(flooring_fn=flooring_fn, rng=self.rng)

    def _init_psdtf(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        r"""Initialize PSDTF.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.
            rng (numpy.random.Generator, optional):
                Random number generator. If ``None`` is given,
                ``np.random.default_rng()`` is used.
                Default: ``None``.
        """
        n_basis = self.n_basis
        n_sources = self.n_sources
        n_bins, n_frames = self.n_bins, self.n_frames

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if rng is None:
            rng = np.random.default_rng()

        if not hasattr(self, "basis"):
            # should be positive semi-definite
            eye = np.eye(n_bins, dtype=np.complex128)
            rand = rng.random((n_sources, n_basis, n_bins))
            T = rand[..., np.newaxis] * eye
        else:
            # To avoid overwriting.
            T = self.basis.copy()

        if not hasattr(self, "activation"):
            V = rng.random((n_sources, n_basis, n_frames))
            V = flooring_fn(V)
        else:
            # To avoid overwriting.
            V = self.activation.copy()

        self.basis, self.activation = T, V

        if self.source_normalization:
            self.normalize_psdtf()

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

    def reconstruct_psdtf(
        self,
        basis: np.ndarray,
        activation: np.ndarray,
        axis1: int = -2,
        axis2: int = -1,
    ) -> np.ndarray:
        r"""Reconstruct PSDTF.

        Args:
            basis (numpy.ndarray):
                Basis matrix.
                The shape is (n_sources, n_basis, n_bins, n_bins) if ``axis1=-1`` and ``axis2=-2``.
                Otherwise, (n_sources, n_bins, n_bins, n_basis).
            activation (numpy.ndarray):
                Activation matrix.
                The shape is (n_sources, n_basis, n_frames).
            axis1 (int):
                First axis of covariance matrix. Default: ``-2``.
            axis2 (int):
                Second axis of covariance matrix. Default: ``-1``.

        Returns:
            numpy.ndarray of reconstructed PSDTF.
            The shape is (n_sources, n_frames, n_bins, n_bins).
        """
        T, V = basis, activation
        n_dims = T.ndim

        axis1 = n_dims + axis1 if axis1 < 0 else axis1
        axis2 = n_dims + axis2 if axis2 < 0 else axis2

        assert (axis1 == 1 and axis2 == 2) or (axis1 == 2 and axis2 == 3)

        if axis1 == 1 and axis2 == 2:
            T = T.transpose(0, 3, 1, 2)

        R = np.sum(T[:, :, np.newaxis, :, :] * V[:, :, :, np.newaxis, np.newaxis], axis=1)
        R = to_psd(R, axis1=2, axis2=3)

        return R

    def update_once(self) -> None:
        r"""Update demixing filters once."""
        raise NotImplementedError("Implement 'update_once' method.")

    def normalize_psdtf(self) -> None:
        r"""Normalize PSDTF parameters."""
        source_normalization = self.source_normalization
        T, V = self.basis, self.activation

        assert source_normalization, "Set source_normalization."

        trace = np.trace(T, axis1=-2, axis2=-1).real
        T = T / trace[:, :, np.newaxis, np.newaxis]
        V = V * trace[:, :, np.newaxis]

        self.basis, self.activation = T, V

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        Returns:
            Computed loss.
        """
        raise NotImplementedError("Implement 'compute_loss' method.")

    def compute_logdet(self, demix_filter: np.ndarray) -> np.ndarray:
        r"""Compute log-determinant of demixing filter

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
            scale_restoration = PROJECTION_BACK_KEYWORDS[0]

        if scale_restoration in PROJECTION_BACK_KEYWORDS:
            self.apply_projection_back()
        elif scale_restoration in MINIMAL_DISTORTION_PRINCIPLE_KEYWORDS:
            self.apply_minimal_distortion_principle()
        else:
            raise ValueError("{} is not supported for scale restoration.".format(scale_restoration))

    def apply_projection_back(self) -> None:
        r"""Apply projection back technique to estimated spectrograms."""
        assert self.scale_restoration, "Set self.scale_restoration=True."

        X, W = self.input, self.demix_filter
        W_scaled = projection_back(W, reference_id=self.reference_id)
        Y_scaled = self.separate(X, demix_filter=W_scaled)

        self.output, self.demix_filter = Y_scaled, W_scaled

    def apply_minimal_distortion_principle(self) -> None:
        r"""Apply minimal distortion principle to estimated spectrograms."""
        assert self.scale_restoration, "Set self.scale_restoration=True."

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        Y_scaled = minimal_distortion_principle(Y, reference=X, reference_id=self.reference_id)
        X = X.transpose(1, 0, 2)
        Y = Y_scaled.transpose(1, 0, 2)
        X_Hermite = X.transpose(0, 2, 1).conj()
        W_scaled = Y @ X_Hermite @ np.linalg.inv(X @ X_Hermite)

        self.output, self.demix_filter = Y_scaled, W_scaled


class BlockDecompositionIPSDTABase(IPSDTABase):
    r"""Base class of independent positive semidefinite tensor analysis (IPSDTA) \
    using block decomposition of bases.

    Args:
        n_basis (int):
            Number of PSDTF bases.
        n_blocks (int):
            Number of sub-blocks.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back`` explicitly.
            Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.
        rng (numpy.random.Generator, optioinal):
            Random number generator. This is mainly used to randomly initialize PSDTF.
            If ``None`` is given, ``np.random.default_rng()`` is used.
            Default: ``None``.
    """

    def __init__(
        self,
        n_basis: int,
        n_blocks: int,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[
                Callable[["BlockDecompositionIPSDTABase"], None],
                List[Callable[["BlockDecompositionIPSDTABase"], None]],
            ]
        ] = None,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            n_basis=n_basis,
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            scale_restoration=scale_restoration,
            record_loss=record_loss,
            reference_id=reference_id,
            rng=rng,
        )

        self.n_blocks = n_blocks

    def __repr__(self) -> str:
        s = "IPSDTA("
        s += "n_basis={n_basis}"
        s += ", n_blocks={n_blocks}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def _reset(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
        **kwargs,
    ) -> None:
        r"""Reset attributes by given keyword arguments.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.
            kwargs:
                Keyword arguments to set as attributes of IPSDTA.
        """
        assert self.input is not None, "Specify data!"

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

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

        self._init_block_decomposition_psdtf(flooring_fn=flooring_fn, rng=self.rng)

    def _init_block_decomposition_psdtf(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        r"""Initialize PSDTF using block decomposition of bases.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.
            rng (numpy.random.Generator, optional):
                Random number generator. If ``None`` is given,
                ``np.random.default_rng()`` is used.
                Default: ``None``.
        """
        n_basis = self.n_basis
        n_sources = self.n_sources
        n_bins, n_frames = self.n_bins, self.n_frames
        n_blocks = self.n_blocks
        n_remains = self.n_remains

        n_neighbors = n_bins // n_blocks

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if rng is None:
            rng = np.random.default_rng()

        if not hasattr(self, "basis"):
            # should be positive semi-definite
            eye = np.eye(n_neighbors, dtype=np.complex128)
            rand = rng.random((n_sources, n_basis, n_blocks - n_remains, n_neighbors))
            T = rand[..., np.newaxis] * eye

            if n_remains > 0:
                eye = np.eye(n_neighbors + 1, dtype=np.complex128)
                rand = rng.random((n_sources, n_basis, n_remains, n_neighbors + 1))
                T_high = rand[..., np.newaxis] * eye

                T = T, T_high
        else:
            # To avoid overwriting.
            if n_remains > 0:
                T_low, T_high = self.basis
                T = T_low.copy(), T_high.copy()
            else:
                T = self.basis.copy()

        if not hasattr(self, "activation"):
            V = rng.random((n_sources, n_basis, n_frames))
            V = flooring_fn(V)
        else:
            # To avoid overwriting.
            V = self.activation.copy()

        self.basis, self.activation = T, V

        if self.source_normalization:
            self.normalize_block_decomposition_psdtf()

    @property
    def n_remains(self) -> int:
        if not hasattr(self, "n_bins"):
            raise AttributeError("Since n_bins is not defined, n_remains cannot be computed.")

        return self.n_bins % self.n_blocks

    def reconstruct_block_decomposition_psdtf(
        self, basis: np.ndarray, activation: np.ndarray, axis1: int = -2, axis2: int = -1
    ) -> np.ndarray:
        r"""Reconstruct PSDTF using block decomposition of bases.

        Args:
            basis (numpy.ndarray):
                Basis matrix.
                The shape is (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
                if ``axis1=-1`` and ``axis2=-2``.
                Otherwise, (n_sources, n_blocks, n_neighbors, n_neighbors, n_basis).
            activation (numpy.ndarray):
                Activation matrix.
                The shape is (n_sources, n_basis, n_frames).
            axis1 (int):
                First axis of covariance matrix. Default: ``-2``.
            axis2 (int):
                Second axis of covariance matrix. Default: ``-1``.

        Returns:
            numpy.ndarray of reconstructed PSDTF.
            The shape is (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors).
        """

        def _reconstruct(
            basis: np.ndarray, activation: np.ndarray, axis1: int = -2, axis2: int = -1
        ) -> np.ndarray:
            r"""Reconstruct PSDTF using block decomposition of bases.

            Args:
                basis (numpy.ndarray):
                    Basis matrix.
                    The shape is (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
                    if ``axis1=-1`` and ``axis2=-2``.
                    Otherwise, (n_sources, n_blocks, n_neighbors, n_neighbors, n_basis).
                activation (numpy.ndarray):
                    Activation matrix.
                    The shape is (n_sources, n_basis, n_frames).
                axis1 (int):
                    First axis of covariance matrix. Default: ``-2``.
                axis2 (int):
                    Second axis of covariance matrix. Default: ``-1``.

            Returns:
                numpy.ndarray of reconstructed PSDTF.
                The shape is (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors).
            """
            na = np.newaxis
            T, V = basis, activation
            n_dims = T.ndim

            axis1 = n_dims + axis1 if axis1 < 0 else axis1
            axis2 = n_dims + axis2 if axis2 < 0 else axis2

            assert (axis1 == 2 and axis2 == 3) or (axis1 == 3 and axis2 == 4)

            if axis1 == 2 and axis2 == 3:
                T = T.transpose(0, 4, 1, 2, 3)

            R = np.sum(
                T[:, :, na, :, :, :] * V[:, :, :, na, na, na],
                axis=1,
            )
            R = to_psd(R, axis1=3, axis2=4)

            return R

        if type(basis) is tuple:
            assert self.n_remains > 0, "n_remains is expected to be positive."

            T_low, T_high = basis
            V = activation
            R_low = _reconstruct(T_low, V, axis1=axis1, axis2=axis2)
            R_high = _reconstruct(T_high, V, axis1=axis1, axis2=axis2)
            R = R_low, R_high
        else:
            T = basis
            V = activation
            R = _reconstruct(T, V, axis1=axis1, axis2=axis2)

        return R

    def normalize_block_decomposition_psdtf(self, axis1: int = -2, axis2: int = -1) -> None:
        r"""Normalize PSDTF parameters using block decomposition of bases.

        Args:
            axis1 (int):
                First axis of covariance matrix. Default: ``-2``.
            axis2 (int):
                Second axis of covariance matrix. Default: ``-1``.
        """
        source_normalization = self.source_normalization
        n_remains = self.n_remains
        na = np.newaxis
        T, V = self.basis, self.activation

        assert source_normalization, "Set source_normalization."

        if n_remains > 0:
            T_low, T_high = T
            trace_low = np.trace(T_low, axis1=axis1, axis2=axis2).real
            trace_high = np.trace(T_high, axis1=axis1, axis2=axis2).real
            trace = np.sum(trace_low, axis=-1) + np.sum(trace_high, axis=-1)
            T_low = T_low / trace[:, :, na, na, na]
            T_high = T_high / trace[:, :, na, na, na]
            T = T_low, T_high
        else:
            trace = np.trace(T, axis1=axis1, axis2=axis2).real
            trace = np.sum(trace, axis=-1)
            T = T / trace[:, :, na, na, na]

        V = V * trace[:, :, na]

        self.basis, self.activation = T, V


class GaussIPSDTA(BlockDecompositionIPSDTABase):
    r"""Independent positive semidefinite tensor analysis (IPSDTA) \
    on Gaussian distribution.

    Args:
        n_basis (int):
            Number of PSDTF bases.
        n_blocks (int):
            Number of sub-blocks.
        source_algorithm (str):
            Algorithm for PSDTF updates.
            Choose ``EM``, or ``MM``. Default: ``MM``.
        spatial_algorithm (str):
            Algorithm for demixing filter updates.
            Choose ``FPI``, or ``VCD``. Default: ``VCD``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        source_normalization (bool):
            If ``source_normalization=True``, normalize PSDTF parameters.
            Default: ``True``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back`` explicitly.
            Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.
        rng (numpy.random.Generator, optioinal):
            Random number generator. This is mainly used to randomly initialize PSDTF.
            If ``None`` is given, ``np.random.default_rng()`` is used.
            Default: ``None``.
    """

    def __init__(
        self,
        n_basis: int,
        n_blocks: int,
        source_algorithm: str = "MM",
        spatial_algorithm: str = "VCD",
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[
                Callable[["GaussIPSDTA"], None],
                List[Callable[["GaussIPSDTA"], None]],
            ]
        ] = None,
        source_normalization: Optional[Union[bool, str]] = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            n_basis,
            n_blocks,
            flooring_fn,
            callbacks,
            scale_restoration,
            record_loss,
            reference_id,
            rng,
        )

        assert source_algorithm in source_algorithms, "Not support {}.".format(source_algorithms)
        assert spatial_algorithm in spatial_algorithms, "Not support {}.".format(spatial_algorithms)

        self.source_algorithm = source_algorithm
        self.spatial_algorithm = spatial_algorithm
        self.source_normalization = source_normalization

    def __repr__(self) -> str:
        s = "GaussIPSDTA("
        s += "n_basis={n_basis}"
        s += ", n_blocks={n_blocks}"
        s += ", source_algorithm={source_algorithm}"
        s += ", spatial_algorithm={spatial_algorithm}"
        s += ", source_normalization={source_normalization}"
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
                Keyword arguments to set as attributes of IPSDTA.
        """
        super()._reset(**kwargs)

        if self.spatial_algorithm == "FPI":
            if not hasattr(self, "fixed_point"):
                n_sources = self.n_sources
                n_bins = self.n_bins

                self.fixed_point = np.ones((n_sources, n_bins), dtype=np.complex128)
            else:
                self.fixed_point = self.fixed_point.copy()

            raise NotImplementedError("IPSDTA with fixed-point iteration is not supported.")

    def update_once(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update PSDTF parameters and demixing filters once.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        self.update_source_model(flooring_fn=flooring_fn)
        self.update_spatial_model(flooring_fn=flooring_fn)

    def update_source_model(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update PSDTF basis matrices and activations.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.source_algorithm == "MM":
            self.update_source_model_mm(flooring_fn=flooring_fn)
        else:
            raise NotImplementedError("Not support {}.".format(self.source_algorithm))

        if self.source_normalization:
            self.normalize_block_decomposition_psdtf()

    def update_source_model_mm(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update PSDTF basis matrices and activations by MM algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        self.update_basis_mm(flooring_fn=flooring_fn)
        self.update_activation_mm()

    def update_basis_mm(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update PSDTF basis matrices by MM algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """
        n_sources = self.n_sources
        n_frames = self.n_frames

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        def _update_basis_mm(
            basis: np.ndarray, activation: np.ndarray, separated: np.ndarray = None
        ) -> np.ndarray:
            r"""
            Args:
                basis: (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
                activation: (n_sources, n_basis, n_frames)
                separated: (n_sources, n_blocks, n_neighbors, n_frames)

            Returns:
                numpy.ndarray of updated basis matrix.
            """
            T, V = basis, activation
            Y = separated
            na = np.newaxis

            R = self.reconstruct_block_decomposition_psdtf(T, V)
            R_inverse = np.linalg.inv(R)
            Y = Y.transpose(0, 3, 1, 2)
            YY_Hermite = Y[:, :, :, :, na] @ Y[:, :, :, na, :].conj()
            RYYR = R_inverse @ YY_Hermite @ R_inverse

            P = np.mean(
                V[:, :, :, na, na, na] * R_inverse[:, na, :, :, :, :],
                axis=2,
            )
            Q = np.mean(
                V[:, :, :, na, na, na] * RYYR[:, na, :, :, :, :],
                axis=2,
            )
            TQT = T @ Q @ T

            P = to_psd(P, flooring_fn=flooring_fn)
            TQT = to_psd(TQT, flooring_fn=flooring_fn)

            # geometric mean of P^(-1) and TQT
            T = gmeanmh(P, TQT, type=2)
            T = to_psd(T, flooring_fn=flooring_fn)

            return T

        n_bins = self.n_bins
        n_blocks = self.n_blocks
        n_remains = self.n_remains
        n_neighbors = n_bins // n_blocks

        X, W = self.input, self.demix_filter
        T, V = self.basis, self.activation
        Y = self.separate(X, demix_filter=W)

        if n_remains > 0:
            T_low, T_high = T
            Y_low, Y_high = np.split(Y, [(n_blocks - n_remains) * n_neighbors], axis=1)
            Y_low = Y_low.reshape(n_sources, n_blocks - n_remains, n_neighbors, n_frames)
            Y_high = Y_high.reshape(n_sources, n_remains, n_neighbors + 1, n_frames)

            T_low = _update_basis_mm(T_low, V, separated=Y_low)
            T_high = _update_basis_mm(T_high, V, separated=Y_high)
            T = T_low, T_high
        else:
            Y = Y.reshape(n_sources, n_blocks, n_neighbors, n_frames)
            T = _update_basis_mm(T, V, separated=Y)

        self.basis = T

    def update_activation_mm(self) -> None:
        r"""Update PSDTF activations by MM algorithm."""

        def _compute_traces(
            basis: np.ndarray, activation: np.ndarray, separated: np.ndarray = None
        ) -> Tuple[np.ndarray, np.ndarray]:
            r"""
            Args:
                basis: (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
                activation: (n_sources, n_basis, n_frames)
                separated: (n_sources, n_blocks, n_neighbors, n_frames)

            Returns:
                Tuple of numerator and denominator.
                Type of each item is ``numpy.ndarray``.
            """
            T, V = basis, activation
            Y = separated
            na = np.newaxis

            R = self.reconstruct_block_decomposition_psdtf(T, V)
            R_inverse = np.linalg.inv(R)
            Y = Y.transpose(0, 3, 1, 2)
            YY_Hermite = Y[:, :, :, :, na] @ Y[:, :, :, na, :].conj()
            RYYR = R_inverse @ YY_Hermite @ R_inverse

            num = np.trace(RYYR[:, na, :] @ T[:, :, na], axis1=-2, axis2=-1)
            denom = np.trace(R_inverse[:, na, :] @ T[:, :, na], axis1=-2, axis2=-1)
            num = np.real(num).sum(axis=-1)
            denom = np.real(denom).sum(axis=-1)

            return num, denom

        n_sources = self.n_sources
        n_bins, n_frames = self.n_bins, self.n_frames
        n_blocks = self.n_blocks
        n_remains = self.n_remains
        n_neighbors = n_bins // n_blocks

        X, W = self.input, self.demix_filter
        T, V = self.basis, self.activation
        Y = self.separate(X, demix_filter=W)

        if n_remains > 0:
            T_low, T_high = T
            Y_low, Y_high = np.split(Y, [(n_blocks - n_remains) * n_neighbors], axis=1)
            Y_low = Y_low.reshape(n_sources, n_blocks - n_remains, n_neighbors, n_frames)
            Y_high = Y_high.reshape(n_sources, n_remains, n_neighbors + 1, n_frames)

            num_low, denom_low = _compute_traces(T_low, V, separated=Y_low)
            num_high, denom_high = _compute_traces(T_high, V, separated=Y_high)

            num = num_low + num_high
            denom = denom_low + denom_high
        else:
            Y = Y.reshape(n_sources, n_blocks, n_neighbors, n_frames)
            num, denom = _compute_traces(T, V, separated=Y)

        self.activation = V * np.sqrt(num / denom)

    def update_spatial_model(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update demixing filters once.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.spatial_algorithm == "VCD":
            self.update_spatial_model_vcd(flooring_fn=flooring_fn)
        else:
            raise NotImplementedError("Not support {}.".format(self.spatial_algorithm))

    def update_spatial_model_vcd(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update demixing filters once by VCD.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        def _update(input: np.ndarray, demix_filter: np.ndarray, covariance: np.ndarray):
            r"""
            Args:
                input (np.ndarray):
                    Mixture spectrogram.
                    The shape is (n_channels, n_blocks, n_neighbors, n_frames).
                demix_filter (np.ndarray):
                    Demixing filter split by frequnecy bands.
                    The shape is (n_blocks, n_neighbors, n_sources, n_channels).
                covariance (np.ndarray):
                    Rconstructed PSDTF.
                    The shape is (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors).

            Returns:
                np.ndarray of demixing filters after update.
            """
            X, W = input, demix_filter
            R = covariance

            XX = X[:, na, :, :, na] * X[na, :, :, na, :].conj()
            XX = XX.transpose(2, 3, 4, 0, 1, 5)

            R_inverse = np.linalg.inv(R)
            R_inverse = R_inverse.transpose(2, 4, 3, 0, 1)

            RXX = np.mean(R_inverse[:, :, :, :, na, na] * XX[:, :, :, na, :, :], axis=-1)

            W = update_by_block_decomposition_vcd(
                W, weighted_covariance=RXX, singular_fn=lambda x: np.abs(x) < flooring_fn(0)
            )

            return W

        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins, n_frames = self.n_bins, self.n_frames
        n_blocks = self.n_blocks
        n_remains = self.n_remains
        na = np.newaxis

        n_neighbors = n_bins // n_blocks

        X, W = self.input, self.demix_filter
        T, V = self.basis, self.activation

        R = self.reconstruct_block_decomposition_psdtf(T, V)

        if n_remains > 0:
            X_low, X_high = np.split(X, [(n_blocks - n_remains) * n_neighbors], axis=1)
            W_low, W_high = np.split(W, [(n_blocks - n_remains) * n_neighbors], axis=0)
            R_low, R_high = R

            # Lower frequency
            X_low = X_low.reshape(n_channels, n_blocks - n_remains, n_neighbors, n_frames)
            W_low = W_low.reshape(n_blocks - n_remains, n_neighbors, n_sources, n_channels)
            W_low = _update(X_low, demix_filter=W_low, covariance=R_low)

            # Higher frequency
            X_high = X_high.reshape(n_channels, n_remains, n_neighbors + 1, n_frames)
            W_high = W_high.reshape(n_remains, n_neighbors + 1, n_sources, n_channels)
            W_high = _update(X_high, demix_filter=W_high, covariance=R_high)

            W_low = W_low.reshape((n_blocks - n_remains) * n_neighbors, n_sources, n_channels)
            W_high = W_high.reshape(n_remains * (n_neighbors + 1), n_sources, n_channels)
            W = np.concatenate([W_low, W_high], axis=0)
        else:
            X = X.reshape(n_channels, n_blocks, n_neighbors, n_frames)
            W = W.reshape(n_blocks, n_neighbors, n_sources, n_channels)
            W = _update(X, demix_filter=W, covariance=R)
            W = W.reshape(n_blocks * n_neighbors, n_sources, n_channels)

        self.demix_filter = W

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        Returns:
            Computed loss.
        """

        def _compute_block_decomposition_loss(
            separated: np.ndarray, demix_filter: np.ndarray, covariance: np.ndarray
        ) -> float:
            r"""
            Args:
                separated (np.ndarray):
                    Separated signal with shape of (n_sources, n_frames, n_blocks, n_neighbors).
                demix_filter (np.ndarray):
                    Demixing filters with shape of (n_blocks, n_neighbors, n_sources, n_channels).
                covariance:
                    Covariance matrix with shape of
                    (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors).
            """
            Y, W = separated, demix_filter
            R = covariance

            n_sources, n_frames, n_blocks, n_neighbors = Y.shape

            Y = Y.reshape(n_sources, n_frames, n_blocks, n_neighbors, 1)
            R_inverse = np.linalg.inv(R)
            Y_Hermite = np.swapaxes(Y, 3, 4).conj()
            YRY = np.sum(Y_Hermite @ R_inverse @ Y, axis=(0, 2, 3, 4))
            YRY = np.real(YRY)
            YRY = np.maximum(YRY, 0)
            _, logdetR = np.linalg.slogdet(R)
            logdetR = logdetR.sum(axis=(0, 2))
            logdetW = self.compute_logdet(W)

            loss = np.mean(YRY + logdetR, axis=0) - 2 * logdetW.sum(axis=(0, 1))
            loss = loss.item()

            return loss

        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins, n_frames = self.n_bins, self.n_frames
        n_blocks = self.n_blocks
        n_remains = self.n_remains

        n_neighbors = n_bins // n_blocks

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        Y = Y.transpose(0, 2, 1)
        T, V = self.basis, self.activation

        R = self.reconstruct_block_decomposition_psdtf(T, V)

        if n_remains > 0:
            Y_low, Y_high = np.split(Y, [(n_blocks - n_remains) * n_neighbors], axis=2)
            W_low, W_high = np.split(W, [(n_blocks - n_remains) * n_neighbors], axis=0)
            R_low, R_high = R

            Y_low = Y_low.reshape(n_sources, n_frames, (n_blocks - n_remains), n_neighbors)
            Y_high = Y_high.reshape(n_sources, n_frames, n_remains, n_neighbors + 1)
            W_low = W_low.reshape((n_blocks - n_remains), n_neighbors, n_sources, n_channels)
            W_high = W_high.reshape(n_remains, n_neighbors + 1, n_sources, n_channels)

            loss_low = _compute_block_decomposition_loss(
                Y_low, demix_filter=W_low, covariance=R_low
            )
            loss_high = _compute_block_decomposition_loss(
                Y_high, demix_filter=W_high, covariance=R_high
            )

            loss = loss_low + loss_high
        else:
            Y = Y.reshape(n_sources, n_frames, n_blocks, n_neighbors)
            W = W.reshape(n_blocks, n_neighbors, n_sources, n_channels)

            loss = _compute_block_decomposition_loss(Y, demix_filter=W, covariance=R)

        return loss


class TIPSDTA(BlockDecompositionIPSDTABase):
    r"""Independent positive semidefinite tensor analysis (IPSDTA) \
    on Student's t distribution.

    Args:
        n_basis (int):
            Number of PSDTF bases.
        n_blocks (int):
            Number of sub-blocks.
        dof (float):
            Degree of freedom parameter.
        source_algorithm (str):
            Algorithm for PSDTF updates.
            Only ``MM`` is supported. Default: ``MM``.
        spatial_algorithm (str):
            Algorithm for demixing filter updates.
            Only ``VCD`` is supported. Default: ``VCD``.
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        callbacks (callable or list[callable], optional):
            Callback functions. Each function is called before separation and at each iteration.
            Default: ``None``.
        source_normalization (bool):
            If ``source_normalization=True``, normalize PSDTF parameters.
            Default: ``True``.
        scale_restoration (bool or str):
            Technique to restore scale ambiguity.
            If ``scale_restoration=True``, the projection back technique is applied to
            estimated spectrograms. You can also specify ``projection_back`` explicitly.
            Default: ``True``.
        record_loss (bool):
            Record the loss at each iteration of the update algorithm if ``record_loss=True``.
            Default: ``True``.
        reference_id (int):
            Reference channel for projection back.
            Default: ``0``.
        rng (numpy.random.Generator, optioinal):
            Random number generator. This is mainly used to randomly initialize PSDTF.
            If ``None`` is given, ``np.random.default_rng()`` is used.
            Default: ``None``.
    """

    def __init__(
        self,
        n_basis: int,
        n_blocks: int,
        dof: float,
        source_algorithm: str = "MM",
        spatial_algorithm: str = "VCD",
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
            max_flooring, eps=EPS
        ),
        callbacks: Optional[
            Union[
                Callable[["GaussIPSDTA"], None],
                List[Callable[["GaussIPSDTA"], None]],
            ]
        ] = None,
        source_normalization: Optional[Union[bool, str]] = True,
        scale_restoration: Union[bool, str] = True,
        record_loss: bool = True,
        reference_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            n_basis,
            n_blocks,
            flooring_fn,
            callbacks,
            scale_restoration,
            record_loss,
            reference_id,
            rng,
        )

        assert source_algorithm in source_algorithms, "Not support {}.".format(source_algorithm)
        assert spatial_algorithm in spatial_algorithms, "Not support {}.".format(spatial_algorithm)

        self.dof = dof
        self.source_algorithm = source_algorithm
        self.source_normalization = source_normalization
        self.spatial_algorithm = spatial_algorithm

    def __repr__(self) -> str:
        s = "TIPSDTA("
        s += "n_basis={n_basis}"
        s += ", n_blocks={n_blocks}"
        s += ", dof={dof}"
        s += ", source_algorithm={source_algorithm}"
        s += ", spatial_algorithm={spatial_algorithm}"
        s += ", source_normalization={source_normalization}"
        s += ", scale_restoration={scale_restoration}"
        s += ", record_loss={record_loss}"

        if self.scale_restoration:
            s += ", reference_id={reference_id}"

        s += ")"

        return s.format(**self.__dict__)

    def update_once(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update PSDTF parameters and demixing filters once.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        self.update_source_model(flooring_fn=flooring_fn)
        self.update_spatial_model(flooring_fn=flooring_fn)

    def update_source_model(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update PSDTF basis matrices and activations.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.source_algorithm == "MM":
            self.update_source_model_mm(flooring_fn=flooring_fn)
        else:
            raise NotImplementedError("Not support {}.".format(self.source_algorithm))

        if self.source_normalization:
            self.normalize_block_decomposition_psdtf()

    def update_source_model_mm(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update PSDTF basis matrices and activations by MM algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        self.update_basis_mm(flooring_fn=flooring_fn)
        self.update_activation_mm()

    def update_basis_mm(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update PSDTF basis matrices by MM algorithm.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """
        n_sources = self.n_sources
        n_frames = self.n_frames

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        def _quadratic(Y: np.ndarray, R: np.ndarray) -> np.ndarray:
            r"""
            Args:
                Y (np.ndarray):
                    Separated spectrams with shape of
                    (n_sources, n_blocks, n_neighbors, n_frames).
                R (np.ndarray):
                    Covariance matrix with shape of
                    (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors).

            Returns:
                Quadratic forms with shape of (n_sources, n_frames).
            """
            Y = Y.transpose(0, 3, 1, 2)
            R_inverse = np.linalg.inv(R)

            YRY = quadratic(Y, R_inverse)
            YRY = np.real(YRY)
            YRY = np.maximum(YRY, 0)
            YRY = YRY.sum(axis=-1)

            return YRY

        def _update_basis_mm(
            basis: np.ndarray,
            activation: np.ndarray,
            separated: np.ndarray = None,
            weight: np.ndarray = None,
        ) -> np.ndarray:
            r"""
            Args:
                basis: (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
                activation: (n_sources, n_basis, n_frames)
                separated: (n_sources, n_blocks, n_neighbors, n_frames)
                weight: (n_sources, n_frames)

            Returns:
                numpy.ndarray of updated basis matrix.
            """
            T, V = basis, activation
            Y = separated
            pi = weight
            na = np.newaxis

            R = self.reconstruct_block_decomposition_psdtf(T, V)
            R_inverse = np.linalg.inv(R)
            Y = Y.transpose(0, 3, 1, 2)
            YY_Hermite = Y[:, :, :, :, na] @ Y[:, :, :, na, :].conj()
            RYYR = R_inverse @ YY_Hermite @ R_inverse
            piRYYR = pi[:, :, na, na, na] * RYYR

            P = np.mean(
                V[:, :, :, na, na, na] * R_inverse[:, na, :, :, :, :],
                axis=2,
            )
            Q = np.mean(
                V[:, :, :, na, na, na] * piRYYR[:, na, :, :, :, :],
                axis=2,
            )
            Q = to_psd(Q, flooring_fn=flooring_fn)
            Q_sqrt = sqrtmh(Q)

            QTPTQ = Q_sqrt @ T @ P @ T @ Q_sqrt
            QTPTQ = to_psd(QTPTQ, flooring_fn=flooring_fn)
            T = T @ Q_sqrt @ invsqrtmh(QTPTQ, flooring_fn=flooring_fn) @ Q_sqrt @ T
            T = to_psd(T, flooring_fn=flooring_fn)

            return T

        n_bins = self.n_bins
        n_blocks = self.n_blocks
        n_remains = self.n_remains
        n_neighbors = n_bins // n_blocks

        nu = self.dof

        X, W = self.input, self.demix_filter
        T, V = self.basis, self.activation

        Y = self.separate(X, demix_filter=W)
        R = self.reconstruct_block_decomposition_psdtf(T, V)

        if n_remains > 0:
            T_low, T_high = T
            Y_low, Y_high = np.split(Y, [(n_blocks - n_remains) * n_neighbors], axis=1)
            Y_low = Y_low.reshape(n_sources, n_blocks - n_remains, n_neighbors, n_frames)
            Y_high = Y_high.reshape(n_sources, n_remains, n_neighbors + 1, n_frames)
            R_low, R_high = R

            YRY_low = _quadratic(Y_low, R_low)
            YRY_high = _quadratic(Y_high, R_high)

            YRY = YRY_low + YRY_high
            pi = (nu + 2 * n_bins) / (nu + 2 * YRY)

            T_low = _update_basis_mm(T_low, V, separated=Y_low, weight=pi)
            T_high = _update_basis_mm(T_high, V, separated=Y_high, weight=pi)
            T = T_low, T_high
        else:
            Y = Y.reshape(n_sources, n_blocks, n_neighbors, n_frames)
            YRY = _quadratic(Y, R)
            pi = (nu + 2 * n_bins) / (nu + 2 * YRY)

            T = _update_basis_mm(T, V, separated=Y, weight=pi)

        self.basis = T

    def update_activation_mm(self) -> None:
        r"""Update PSDTF activations by MM algorithm."""

        def _quadratic(Y: np.ndarray, R: np.ndarray) -> np.ndarray:
            r"""
            Args:
                Y (np.ndarray):
                    Separated spectrams with shape of
                    (n_sources, n_blocks, n_neighbors, n_frames).
                R (np.ndarray):
                    Covariance matrix with shape of
                    (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors).

            Returns:
                Quadratic forms with shape of (n_sources, n_frames).
            """
            Y = Y.transpose(0, 3, 1, 2)
            R_inverse = np.linalg.inv(R)

            YRY = quadratic(Y, R_inverse)
            YRY = np.real(YRY)
            YRY = np.maximum(YRY, 0)
            YRY = YRY.sum(axis=-1)

            return YRY

        def _compute_traces(
            basis: np.ndarray,
            activation: np.ndarray,
            separated: np.ndarray = None,
            weight: np.ndarray = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
            r"""
            Args:
                basis: (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
                activation: (n_sources, n_basis, n_frames)
                separated: (n_sources, n_blocks, n_neighbors, n_frames)

            Returns:
                Tuple of numerator and denominator.
                Type of each item is ``numpy.ndarray``.
            """
            T, V = basis, activation
            Y = separated.transpose(0, 3, 1, 2)
            pi = weight
            na = np.newaxis

            R = self.reconstruct_block_decomposition_psdtf(T, V)
            R_inverse = np.linalg.inv(R)
            YY_Hermite = Y[:, :, :, :, na] @ Y[:, :, :, na, :].conj()
            RYYR = R_inverse @ YY_Hermite @ R_inverse

            piRYYR = pi[:, :, na, na, na] * RYYR

            num = np.trace(piRYYR[:, na, :] @ T[:, :, na], axis1=-2, axis2=-1)
            denom = np.trace(R_inverse[:, na, :] @ T[:, :, na], axis1=-2, axis2=-1)
            num = np.real(num).sum(axis=-1)
            denom = np.real(denom).sum(axis=-1)

            return num, denom

        n_sources = self.n_sources
        n_bins, n_frames = self.n_bins, self.n_frames
        n_blocks = self.n_blocks
        n_remains = self.n_remains
        n_neighbors = n_bins // n_blocks

        nu = self.dof

        X, W = self.input, self.demix_filter
        T, V = self.basis, self.activation

        Y = self.separate(X, demix_filter=W)
        R = self.reconstruct_block_decomposition_psdtf(T, V)

        if n_remains > 0:
            T_low, T_high = T
            Y_low, Y_high = np.split(Y, [(n_blocks - n_remains) * n_neighbors], axis=1)
            Y_low = Y_low.reshape(n_sources, n_blocks - n_remains, n_neighbors, n_frames)
            Y_high = Y_high.reshape(n_sources, n_remains, n_neighbors + 1, n_frames)
            R_low, R_high = R

            YRY_low = _quadratic(Y_low, R_low)
            YRY_high = _quadratic(Y_high, R_high)

            YRY = YRY_low + YRY_high
            pi = (nu + 2 * n_bins) / (nu + 2 * YRY)

            num_low, denom_low = _compute_traces(T_low, V, separated=Y_low, weight=pi)
            num_high, denom_high = _compute_traces(T_high, V, separated=Y_high, weight=pi)

            num = num_low + num_high
            denom = denom_low + denom_high
        else:
            Y = Y.reshape(n_sources, n_blocks, n_neighbors, n_frames)
            YRY = _quadratic(Y, R)
            pi = (nu + 2 * n_bins) / (nu + 2 * YRY)

            num, denom = _compute_traces(T, V, separated=Y, weight=pi)

        self.activation = V * np.sqrt(num / denom)

    def update_spatial_model(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update demixing filters once.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """
        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        if self.spatial_algorithm == "VCD":
            self.update_spatial_model_vcd(flooring_fn=flooring_fn)
        else:
            raise NotImplementedError("Not support {}.".format(self.spatial_algorithm))

    def update_spatial_model_vcd(
        self,
        flooring_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "self",
    ) -> None:
        r"""Update demixing filters once by VCD.

        Args:
            flooring_fn (callable or str, optional):
                A flooring function for numerical stability.
                This function is expected to return the same shape tensor as the input.
                If you explicitly set ``flooring_fn=None``,
                the identity function (``lambda x: x``) is used.
                If ``self`` is given as str, ``self.flooring_fn`` is used.
                Default: ``self``.

        """

        flooring_fn = choose_flooring_fn(flooring_fn, method=self)

        def _quadratic(Y: np.ndarray, R: np.ndarray) -> np.ndarray:
            r"""
            Args:
                Y (np.ndarray):
                    Separated spectrams with shape of
                    (n_sources, n_blocks, n_neighbors, n_frames).
                R (np.ndarray):
                    Covariance matrix with shape of
                    (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors).

            Returns:
                Quadratic forms with shape of (n_sources, n_frames).
            """
            Y = Y.transpose(2, 3, 0, 1)
            R_inverse = np.linalg.inv(R)

            YRY = quadratic(Y, R_inverse)
            YRY = np.real(YRY)
            YRY = np.maximum(YRY, 0)
            YRY = YRY.sum(axis=-1)

            return YRY

        def _update(
            input: np.ndarray,
            demix_filter: np.ndarray,
            covariance: np.ndarray,
            weight: np.ndarray = None,
        ):
            X, W = input, demix_filter
            R = covariance
            pi = weight

            na = np.newaxis

            XX = X[:, na, :, :, na] * X[na, :, :, na, :].conj()
            XX = XX.transpose(2, 3, 4, 0, 1, 5)

            R_inverse = np.linalg.inv(R)
            R_inverse = R_inverse.transpose(2, 4, 3, 0, 1)
            pi_R_inverse = pi * R_inverse

            RXX = np.mean(pi_R_inverse[:, :, :, :, na, na] * XX[:, :, :, na, :, :], axis=-1)

            W = update_by_block_decomposition_vcd(
                W, weighted_covariance=RXX, singular_fn=lambda x: np.abs(x) < flooring_fn(0)
            )

            return W

        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins, n_frames = self.n_bins, self.n_frames
        n_blocks = self.n_blocks
        n_remains = self.n_remains
        n_neighbors = n_bins // n_blocks

        nu = self.dof

        X, W = self.input, self.demix_filter
        T, V = self.basis, self.activation

        R = self.reconstruct_block_decomposition_psdtf(T, V)

        if n_remains > 0:
            X_low, X_high = np.split(X, [(n_blocks - n_remains) * n_neighbors], axis=1)
            W_low, W_high = np.split(W, [(n_blocks - n_remains) * n_neighbors], axis=0)
            R_low, R_high = R

            # Lower frequency
            X_low = X_low.reshape(n_channels, n_blocks - n_remains, n_neighbors, n_frames)
            W_low = W_low.reshape(n_blocks - n_remains, n_neighbors, n_sources, n_channels)
            Y_low = W_low @ X_low.transpose(1, 2, 0, 3)

            # Higher frequency
            X_high = X_high.reshape(n_channels, n_remains, n_neighbors + 1, n_frames)
            W_high = W_high.reshape(n_remains, n_neighbors + 1, n_sources, n_channels)
            Y_high = W_high @ X_high.transpose(1, 2, 0, 3)

            YRY_low = _quadratic(Y_low, R_low)
            YRY_high = _quadratic(Y_high, R_high)

            YRY = YRY_low + YRY_high
            pi = (nu + 2 * n_bins) / (nu + 2 * YRY)

            W_low = _update(X_low, demix_filter=W_low, covariance=R_low, weight=pi)
            W_high = _update(X_high, demix_filter=W_high, covariance=R_high, weight=pi)

            W_low = W_low.reshape((n_blocks - n_remains) * n_neighbors, n_sources, n_channels)
            W_high = W_high.reshape(n_remains * (n_neighbors + 1), n_sources, n_channels)
            W = np.concatenate([W_low, W_high], axis=0)
        else:
            X = X.reshape(n_channels, n_blocks, n_neighbors, n_frames)
            W = W.reshape(n_blocks, n_neighbors, n_sources, n_channels)
            Y = W @ X.transpose(1, 2, 0, 3)

            YRY = _quadratic(Y, R)
            pi = (nu + 2 * n_bins) / (nu + 2 * YRY)

            W = _update(X, demix_filter=W, covariance=R, weight=pi)
            W = W.reshape(n_blocks * n_neighbors, n_sources, n_channels)

        self.demix_filter = W

    def compute_loss(self) -> float:
        r"""Compute loss :math:`\mathcal{L}`.

        Returns:
            Computed loss.
        """

        def _quadratic(Y: np.ndarray, R: np.ndarray) -> np.ndarray:
            r"""
            Args:
                Y (np.ndarray):
                    Separated spectrams with shape of
                    (n_sources, n_blocks, n_neighbors, n_frames).
                R (np.ndarray):
                    Covariance matrix with shape of
                    (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors).

            Returns:
                Quadratic forms with shape of (n_sources, n_frames).
            """
            Y = Y.transpose(0, 3, 1, 2)
            R_inverse = np.linalg.inv(R)

            YRY = quadratic(Y, R_inverse)
            YRY = np.real(YRY)
            YRY = np.maximum(YRY, 0)
            YRY = YRY.sum(axis=-1)

            return YRY

        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins, n_frames = self.n_bins, self.n_frames

        nu = self.dof

        n_blocks = self.n_blocks
        n_remains = self.n_remains

        n_neighbors = n_bins // n_blocks

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        T, V = self.basis, self.activation

        R = self.reconstruct_block_decomposition_psdtf(T, V)

        if n_remains > 0:
            Y_low, Y_high = np.split(Y, [(n_blocks - n_remains) * n_neighbors], axis=1)
            W_low, W_high = np.split(W, [(n_blocks - n_remains) * n_neighbors], axis=0)
            R_low, R_high = R

            Y_low = Y_low.reshape(n_sources, (n_blocks - n_remains), n_neighbors, n_frames)
            Y_high = Y_high.reshape(n_sources, n_remains, n_neighbors + 1, n_frames)
            W_low = W_low.reshape((n_blocks - n_remains), n_neighbors, n_sources, n_channels)
            W_high = W_high.reshape(n_remains, n_neighbors + 1, n_sources, n_channels)

            YRY_low = _quadratic(Y_low, R_low)
            YRY_high = _quadratic(Y_high, R_high)

            YRY = YRY_low + YRY_high

            loss = np.sum(((nu + 2 * n_bins) / 2) * np.log(1 + (2 / nu) * YRY), axis=0)

            _, logdetR_low = np.linalg.slogdet(R_low)
            logdetR_low = logdetR_low.sum(axis=(0, 2))
            _, logdetR_high = np.linalg.slogdet(R_high)
            logdetR_high = logdetR_high.sum(axis=(0, 2))
            logdetR = logdetR_low + logdetR_high

            logdetW_low = self.compute_logdet(W_low)
            logdetW_high = self.compute_logdet(W_high)

            logdetW = logdetW_low.sum(axis=(0, 1)) + logdetW_high.sum(axis=(0, 1))
        else:
            Y = Y.reshape(n_sources, n_blocks, n_neighbors, n_frames)
            W = W.reshape(n_blocks, n_neighbors, n_sources, n_channels)

            YRY = _quadratic(Y, R)

            loss = np.sum(((nu + 2 * n_bins) / 2) * np.log(1 + (2 / nu) * YRY), axis=0)

            _, logdetR = np.linalg.slogdet(R)
            logdetR = logdetR.sum(axis=(0, 2))

            logdetW = self.compute_logdet(W)
            logdetW = logdetW.sum(axis=(0, 1))

        loss = np.mean(loss + logdetR, axis=0) - 2 * logdetW
        loss = loss.item()

        return loss
