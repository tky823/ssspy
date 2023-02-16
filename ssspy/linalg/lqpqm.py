from typing import Optional

import numpy as np

from .cubic import cbrt


def solve_equation(Phi: np.ndarray, v: np.ndarray, z: np.ndarray):
    r"""Find largest root of :math:`f(\lambda_{in})`, where

    .. math::

        f(\lambda_{in})
        = \lambda_{in}^{2}\sum_{n'}
        \frac{\phi_{inn'}|\tilde{v}_{inn'}|^{2}}{(\lambda_{in}-\phi_{inn'})^{2}}
        - \lambda_{in} + z_{in}

    Args:
        Phi (numpy.ndarray): Eigen values defined in LQPQM of shape (n_bins, n_sources).
        v (numpy.ndarray): Linear term defined in LQPQM of shape (n_bins, n_sources).
        z (numpy.ndarray): Constant term defined in LQPQM of shape (n_bins,).

    Returns:
        numpy.ndarray of largest root of :math:`f(\lambda_{in})`.
        The shape is (n_bins,).

    """
    phi_max = Phi[..., -1]
    v_max = v[..., -1]

    # Find largest root of cubic polynomial for initialization
    A = -(phi_max * np.abs(v_max) ** 2 + 2 * phi_max + z)
    B = (phi_max + 2 * z) * phi_max
    C = -(phi_max**2) * z
    lamb = _find_largest_root(A, B, C)

    lamb = np.maximum(lamb, z)

    # TODO: generalize for-loop
    for _ in range(10):
        f = _fn(lamb, Phi, v, z)
        df = _d_fn(lamb, Phi, v, z)
        mu = lamb - f / df
        lamb = np.where(mu > phi_max, mu, (phi_max + lamb) / 2)

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


def _fn(lamb: np.ndarray, Phi: np.ndarray, f: np.ndarray, z: np.ndarray) -> np.ndarray:
    r"""Compute values of :math:`f(\lambda_{in})`, where

    .. math::

        f(\lambda_{in})
        = \lambda_{in}^{2}\sum_{n'}
        \frac{\phi_{inn'}|\tilde{v}_{inn'}|^{2}}{(\lambda_{in}-\phi_{inn'})^{2}}
        - \lambda_{in} + z_{in}

    Args:
        lamb (numpy.ndarray): Argument of :math:`f(\lambda_{in})` with shape of (n_bins,).
        Phi (numpy.ndarray): Eigen values defined in LQPQM of shape (n_bins, n_sources).
        v (numpy.ndarray): Linear term defined in LQPQM of shape (n_bins, n_sources).
        z (numpy.ndarray): Constant term defined in LQPQM of shape (n_bins,).

    Returns:
        numpy.ndarray of values :math:`f(\lambda_{in})` of shape (n_bins,).

    """
    num = Phi * np.abs(f) ** 2
    denom = (lamb[..., np.newaxis] - Phi) ** 2
    f = lamb**2 * np.sum(num / denom, axis=-1) - lamb + z

    return f


def _d_fn(
    lamb: np.ndarray,
    Phi: np.ndarray,
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
        Phi (numpy.ndarray): Eigen values defined in LQPQM of shape (n_bins, n_sources).
        v (numpy.ndarray): Linear term defined in LQPQM of shape (n_bins, n_sources).
        z (numpy.ndarray, optional): Constant term defined in LQPQM of shape (n_bins,).
            This argument is not used in this funtion.

    Returns:
        numpy.ndarray of values :math:`f'(\lambda_{in})` of shape (n_bins,).

    """
    num = (Phi * np.abs(v)) ** 2
    denom = (lamb[..., np.newaxis] - Phi) ** 3
    df = -2 * lamb * np.sum(num / denom, axis=-1) - 1

    return df
