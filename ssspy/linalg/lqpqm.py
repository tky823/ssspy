import numpy as np

from .cubic import cbrt


def find_largest_root(A: np.ndarray, B: np.ndarray, C: np.ndarray):
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
        there exists three real solutions: :math:`t=u+\frac{p}{3u}`,
        :math:`x=u\omega+\frac{p\omega^{*}}{3u}`, and
        :math:`x=u\omega^{*}+\frac{p\omega}{3u}`, where

        .. math::

            u
            &=\sqrt[3]{-\frac{q}{2}+\sqrt{\frac{q^{2}}{4} + \frac{p^{3}}{27}}}, \\
            \omega
            &=\frac{-1+j\sqrt{3}}{2}.

        Otherwise, :math:`t=u+v` is a unique real solution.

    """
    P = -(A**2) / 3 + B
    Q = (2 * A**3) / 27 - (A * B) / 3 + C

    omega = (-1 + 1j * np.sqrt(3)) / 2
    omega_conj = (-1 - 1j * np.sqrt(3)) / 2

    discriminant = (Q / 2) ** 2 + (P / 3) ** 3
    discriminant = discriminant.astype(np.complex128)
    U = cbrt(-Q / 2 + np.sqrt(discriminant))
    V = -P / (3 * U)

    X1 = U + V
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
