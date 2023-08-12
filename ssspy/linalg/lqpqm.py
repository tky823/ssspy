import functools
import warnings
from typing import Callable, Optional, Union

import numpy as np

from ..special.flooring import identity, max_flooring
from .cubic import cbrt

EPS = 1e-10


def lqpqm2(
    H: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
    singular_fn: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = "flooring",
    max_iter: int = 10,
) -> None:
    r"""Solve of log-quadratically penelized quadratic minimization (type 2).

    .. math::

        \check{\boldsymbol{q}}_{in}
        = \min_{\check{\boldsymbol{q}}_{in}}
        ~~\check{\boldsymbol{q}}_{in}^{\mathsf{H}}\check{\boldsymbol{q}}_{in}
        - \log\left((\check{\boldsymbol{q}}_{in}+\boldsymbol{v}_{in})^{\mathsf{H}}
        \boldsymbol{H}_{in}(\check{\boldsymbol{q}}_{in}+\boldsymbol{v}_{in})
        + z_{in}
        \right)

    Args:
        H (numpy.ndarray): Positive semidefinite matrices of shape
            (n_bins, n_sources - 1, n_sources - 1).
        v (numpy.ndarray): Linear terms in LQPQM of shape (n_bins, n_sources - 1).
        z (numpy.ndarray): Constant terms in LQPQM of shape (n_bins,).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        singular_fn (callable, optional):
            A flooring function to return singular condition.
            This function is expected to return the same shape bool tensor as the input.
            If ``singular_fn=None``, ``lambda x: x == 0`` is used.
            Default: ``flooring``.
        max_iter (int):
            Maximum number of Newton-Raphson method. Default: ``10``.

    Returns:
        np.ndarray: Solutions of LQPQM type-2 of shape (n_bins, n_sources - 1).

    """
    if flooring_fn is None:
        flooring_fn = identity

    if singular_fn is None:

        def _is_zero(x: np.ndarray) -> np.ndarray:
            return x == 0

        singular_fn = _is_zero
    elif singular_fn == "flooring":

        def _is_lower_than_floor(x: np.ndarray) -> np.ndarray:
            return x < flooring_fn(0)

        singular_fn = _is_lower_than_floor
    else:
        assert callable(singular_fn), "singular_fn should be callable."

    phi, sigma = np.linalg.eigh(H)
    norm = np.linalg.norm(v, axis=-1)
    is_singular = singular_fn(norm)

    # when v = 0
    phi_singular = phi[is_singular]
    sigma_singular = sigma[is_singular]
    z_singular = z[is_singular]

    phi_max_singular = phi_singular[:, -1]
    sigma_max_singular = sigma_singular[:, -1]
    lamb_singular = np.maximum(z_singular, phi_max_singular)
    scale = (lamb_singular - z_singular) / phi_max_singular
    scale = np.maximum(scale, 0)
    scale = np.sqrt(scale)
    y_singular = scale[..., np.newaxis] * sigma_max_singular

    # when v != 0
    phi_non_singular = phi[~is_singular]
    sigma_non_singular = sigma[~is_singular]
    v_non_singular = v[~is_singular]
    z_non_singular = z[~is_singular]

    v_tilde_non_singular = np.sum(
        sigma_non_singular.conj() * v_non_singular[:, :, np.newaxis], axis=-2
    )
    lamb_non_singular = solve_equation(
        phi_non_singular,
        v_tilde_non_singular,
        z_non_singular,
        flooring_fn=flooring_fn,
        max_iter=max_iter,
        normalization=True,
    )

    num = phi_non_singular * v_tilde_non_singular
    denom = lamb_non_singular[..., np.newaxis] - phi_non_singular
    v_nonsingular = num / denom
    y_non_singular = np.sum(sigma_non_singular * v_nonsingular[:, np.newaxis, :], axis=-1)

    y = np.zeros_like(v)
    y[is_singular] = y_singular
    y[~is_singular] = y_non_singular

    return y


def solve_equation(
    phi: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = functools.partial(
        max_flooring, eps=EPS
    ),
    max_iter: int = 10,
    normalization: bool = True,
):
    r"""Find largest root of :math:`f(\lambda_{in})`, where

    .. math::

        f(\lambda_{in})
        = \lambda_{in}^{2}\sum_{n'}
        \frac{\phi_{inn'}|\tilde{v}_{inn'}|^{2}}{(\lambda_{in}-\phi_{inn'})^{2}}
        - \lambda_{in} + z_{in}

    Args:
        phi (numpy.ndarray): Eigen values defined in LQPQM of shape (n_bins, n_sources).
        v (numpy.ndarray): Linear term defined in LQPQM of shape (n_bins, n_sources).
        z (numpy.ndarray): Constant term defined in LQPQM of shape (n_bins,).
        flooring_fn (callable, optional):
            A flooring function for numerical stability.
            This function is expected to return the same shape tensor as the input.
            If you explicitly set ``flooring_fn=None``,
            the identity function (``lambda x: x``) is used.
            Default: ``functools.partial(max_flooring, eps=1e-10)``.
        max_iter (int): Maximum iteration of Newton-Raphson method. Default: ``10``.
        normalization (bool): If ``True``, coefficients are normalized by ``phi_max``.

    Returns:
        numpy.ndarray of largest root of :math:`f(\lambda_{in})`.
        The shape is (n_bins,).

    """
    if flooring_fn is None:
        flooring_fn = identity

    n_bins, n_sources = phi.shape

    non_zero_mask = phi * np.abs(v) ** 2 >= flooring_fn(0)
    phi = non_zero_mask * phi
    v = non_zero_mask * v

    max_index = np.argmax(phi, axis=-1) + np.arange(0, n_bins * n_sources, n_sources)
    phi_flatten = phi.flatten()
    v_flatten = v.flatten()
    phi_max = phi_flatten[max_index]
    v_max = v_flatten[max_index]
    phi_max = flooring_fn(phi_max)

    if normalization:
        phi_max_original = phi_max
        phi = phi / phi_max[:, np.newaxis]
        v = v / phi_max[:, np.newaxis]
        v_max = v_max / phi_max
        z = z / phi_max
        phi_max = phi_max / phi_max  # i.e. phi_max = 1
    else:
        phi_max_original = None

    # Find largest root of cubic polynomial for initialization
    A = -(phi_max * np.abs(v_max) ** 2 + 2 * phi_max + z)
    B = (phi_max + 2 * z) * phi_max
    C = -(phi_max**2) * z
    lamb = _find_largest_root(A, B, C)

    is_valid = lamb > phi_max
    lamb[~is_valid] = phi_max[~is_valid] + flooring_fn(0)
    lamb = np.maximum(lamb, z)

    for iter_idx in range(max_iter):
        f = _fn(lamb, phi, v, z)
        is_convergence = np.abs(f) <= flooring_fn(0)

        if np.all(is_convergence):
            break

        df = _d_fn(lamb, phi, v, z)
        mu = lamb - f / df
        lamb = np.where(mu > phi_max, mu, (phi_max + lamb) / 2)

    if iter_idx == max_iter - 1:
        f = _fn(lamb, phi, v, z)
        is_convergence = np.abs(f) <= flooring_fn(0)

        if not np.all(is_convergence):
            warnings.warn(
                f"Newton-Raphson method did not converge in {max_iter} iterations.", UserWarning
            )

    if normalization:
        lamb = lamb * phi_max_original

    return lamb


def _find_largest_root(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    r"""Find largest (real) roots of the following cubic equations:

    .. math::

        x^{3} + Ax^{2} + Bx + C = 0.

    Args:
        A (numpy.ndarray): Coefficients of quadratic terms with shape of (\*).
        B (numpy.ndarray): Coefficients of linear terms with shape of (\*).
        C (numpy.ndarray): Coefficients of constant terms with shape of (\*).

    Returns:
        numpy.ndarray of largest (real) roots.

    .. note::

        :math:`x^{3} + Ax^{2} + Bx + C = 0` can be transformed into
        :math:`t^{3} + pt + q = 0` by :math:`t=x+\frac{A}{3}`.
        When :math:`p<0` and :math:`\frac{q^{2}}{4}+\frac{p^{3}}{27}\leq 0`,
        there exists three real solutions: :math:`t=u-\frac{p}{3u}`,
        :math:`x=u\omega-\frac{p\omega^{*}}{3u}`, and
        :math:`x=u\omega^{*}-\frac{p\omega}{3u}`, where

        .. math::

            u
            &=\sqrt[3]{-\frac{q}{2}+\sqrt{\frac{q^{2}}{4} + \frac{p^{3}}{27}}}, \\
            \omega
            &=\frac{-1+j\sqrt{3}}{2}.

        When :math:`p<0` and :math:`\frac{q^{2}}{4}+\frac{p^{3}}{27}>0`,
        :math:`t=u-p/(3u)` is a unique real solution.
        When :math:`p>0`, :math:`t=u-p/(3u)` is a unique real solution.
        Otherwise (when :math:`p=0`), :math:`t=\sqrt[3]{-q}` is a unique real solution.

    """
    P = -(A**2) / 3 + B
    Q = (2 * A**3) / 27 - (A * B) / 3 + C

    omega = (-1 + 1j * np.sqrt(3)) / 2
    omega_conj = (-1 - 1j * np.sqrt(3)) / 2

    discriminant = (Q / 2) ** 2 + (P / 3) ** 3
    discriminant = discriminant.astype(np.complex128)
    U = cbrt(-Q / 2 + np.sqrt(discriminant))
    # When U = 0, P is always 0 in real coefficients cases.
    is_singular = U == 0
    U = np.where(is_singular, 1, U)
    V = -P / (3 * U)

    X1 = U + V
    X1 = np.where(is_singular, cbrt(-Q), X1)
    X2 = np.real(U * omega + V * omega_conj)
    X3 = np.real(U * omega_conj + V * omega)

    roots = np.stack([X1, X2, X3], axis=-1)
    roots = np.real(roots)

    is_monotonic = P >= 0
    is_unique = np.array([True, False, False])

    imaginary_mask = is_monotonic[..., np.newaxis] & ~is_unique
    roots = np.where(imaginary_mask, -float("inf"), roots)
    imaginary_mask = ~is_monotonic[..., np.newaxis] & ~is_unique
    is_positive = discriminant > 0
    roots = np.where(imaginary_mask & is_positive[..., np.newaxis], -float("inf"), roots)
    root = np.max(roots, axis=-1)
    root = root - A / 3

    return root


def _fn(lamb: np.ndarray, phi: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
    r"""Compute values of :math:`f(\lambda_{in})`, where

    .. math::

        f(\lambda_{in})
        = \lambda_{in}^{2}\sum_{n'}
        \frac{\phi_{inn'}|\tilde{v}_{inn'}|^{2}}{(\lambda_{in}-\phi_{inn'})^{2}}
        - \lambda_{in} + z_{in}

    Args:
        lamb (numpy.ndarray): Argument of :math:`f(\lambda_{in})` with shape of (n_bins,).
        phi (numpy.ndarray): Eigen values defined in LQPQM of shape (n_bins, n_sources).
        v (numpy.ndarray): Linear term defined in LQPQM of shape (n_bins, n_sources).
        z (numpy.ndarray): Constant term defined in LQPQM of shape (n_bins,).

    Returns:
        numpy.ndarray of values :math:`f(\lambda_{in})` of shape (n_bins,).

    """
    num = phi * np.abs(v) ** 2
    denom = (lamb[..., np.newaxis] - phi) ** 2
    f = lamb**2 * np.sum(num / denom, axis=-1) - lamb + z

    return f


def _d_fn(
    lamb: np.ndarray,
    phi: np.ndarray,
    v: np.ndarray,
    z: Optional[np.ndarray] = None,
):
    r"""Compute values of :math:`f'(\lambda_{in})`, where

    .. math::

        f'(\lambda_{in})
        = -2\lambda_{in}\sum_{n'}
        \frac{\phi_{inn'}^{2}|\tilde{v}_{inn'}|^{2}}{(\lambda_{in}-\phi_{inn'})^{3}}
        - 1

    Args:
        lamb (numpy.ndarray): Argument of :math:`f'(\lambda_{in})` with shape of (n_bins,).
        phi (numpy.ndarray): Eigen values defined in LQPQM of shape (n_bins, n_sources).
        v (numpy.ndarray): Linear term defined in LQPQM of shape (n_bins, n_sources).
        z (numpy.ndarray, optional): Constant term defined in LQPQM of shape (n_bins,).
            This argument is not used in this funtion.

    Returns:
        numpy.ndarray of values :math:`f'(\lambda_{in})` of shape (n_bins,).

    """
    num = (phi * np.abs(v)) ** 2
    denom = (lamb[..., np.newaxis] - phi) ** 3
    df = -2 * lamb * np.sum(num / denom, axis=-1) - 1

    return df
