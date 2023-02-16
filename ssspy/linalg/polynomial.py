from typing import Optional

import numpy as np
from numpy.linalg import LinAlgError

from .cubic import cbrt


def solve_cubic(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: Optional[np.ndarray] = None,
    all: bool = True,
) -> np.ndarray:
    r"""Find roots of cubic equations.

    Args:
        A (numpy.ndarray): Coefficients of cubic or quadratic terms.
        B (numpy.ndarray): Coefficients of quadratic or linear terms.
        C (numpy.ndarray): Coefficients of linear or constant terms.
        D (numpy.ndarray, optional): Constant terms.
        all (bool): If ``all=True``, returns all roots. Otherwise, returns one of them.
            Default: ``True``.

    Returns:
        numpy.ndarray: All roots of cuadratic equations of shape (3, \*) if ``all=True``.
            Otherwise, (\*).

    This function solves the following equations if ``D`` is given:

    .. math::

        Ax^{3} + Bx^{2} + Cx + D = 0.

    If ``D`` is not given, solves

    .. math::

        x^{3} + Ax^{2} + Bx + C = 0.

    """
    if D is None:
        P = -(A**2) / 3 + B
        Q = (2 * A**3) / 27 - (A * B) / 3 + C

        X = _find_cubic_roots(P, Q)
        x = X - A / 3

        return x if all else x[0]
    else:
        if np.any(A == 0):
            raise LinAlgError("Coefficients include zero.")

        return solve_cubic(B / A, C / A, D / A, all=all)


def _find_cubic_roots(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    r"""Find roots of the following cubic equations:

    .. math::

        x^{3} + px + q = 0


    Args:
        P (np.ndarray): Coefficients of cubic equation.
        Q (np.ndarray): Coefficients of cubic equation.

    Returns:
        numpy.ndarray of the three roots.
        The shape is (3, \*).

    """
    P = P.astype(np.complex128)
    Q = Q.astype(np.complex128)
    omega = (-1 + 1j * np.sqrt(3)) / 2
    omega_conj = (-1 - 1j * np.sqrt(3)) / 2

    discriminant = (Q / 2) ** 2 + (P / 3) ** 3

    U = cbrt(-Q / 2 + np.sqrt(discriminant))
    # U = 0, when P = 0.
    is_singular = P == 0
    U = np.where(is_singular, 1, U)
    V = -P / (3 * U)

    X1 = U + V
    X1 = np.where(is_singular, cbrt(-Q), X1)
    X2 = U * omega + V * omega_conj
    X2 = np.where(is_singular, X1 * omega, X2)
    X3 = U * omega_conj + V * omega
    X3 = np.where(is_singular, X1 * omega_conj, X3)

    return np.stack([X1, X2, X3], axis=0)
