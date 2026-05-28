"""
Analytical reference solution for the 1D bar problem.

Solves the strong form:
    d/dx(EA * du/dx) + b(x) = 0  on (0, L)
    u(0) = 0                      (fixed end)
    EA * du/dx|_{x=L} = P         (applied tip force)

For constant EA and constant body load b_0, integration gives:
    u(x) = -b_0/(2*EA) * x^2 + (P + b_0*L)/EA * x

This is the closed-form solution we'll use to validate the FEA code.
"""

import numpy as np


def u_exact(x, E, A, L, b0, P):
    """Closed-form displacement field for the 1D bar with constant body load.

    Parameters
    ----------
    x : float or array
        Position(s) along the bar where displacement is evaluated.
    E : float
        Young's modulus.
    A : float
        Cross-sectional area.
    L : float
        Bar length.
    b0 : float
        Uniform body load (force per unit length).
    P : float
        Point load applied at x = L.

    Returns
    -------
    u : float or array
        Displacement at the requested position(s).
    """
    EA = E * A
    return -b0 / (2.0 * EA) * x**2 + (P + b0 * L) / EA * x