"""
Analytical reference for 1D steady-state heat conduction.

Two cases, both derived by hand (Phase 3 notes):

  Case A (no source, Q0 = 0): linear profile
      T(x) = (TL - T0)/L * x + T0

  Case B (uniform source Q0): quadratic profile
      T(x) = -Q0/(2 k A) x^2 + [ (TL - T0)/L + Q0 L /(2 k A) ] x + T0

  (Case B reduces to Case A when Q0 = 0 — a good built-in check.)
"""

import numpy as np


def T_exact(x, k, A, L, T0, TL, Q0):
    """Closed-form temperature profile for the 1D bar.

    Parameters
    ----------
    x : float or array
        Position(s) along the bar.
    k : float
        Thermal conductivity.
    A : float
        Cross-sectional area.
    L : float
        Bar length.
    T0, TL : float
        Prescribed temperatures at x=0 and x=L.
    Q0 : float
        Uniform heat source (0 for Case A).

    Returns
    -------
    T : float or array
    """
    kA = k * A
    return -Q0 / (2.0 * kA) * x**2 + ((TL - T0) / L + Q0 * L / (2.0 * kA)) * x + T0