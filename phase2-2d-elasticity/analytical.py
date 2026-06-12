"""
Analytical reference for the 2D cantilever benchmark.

Euler-Bernoulli beam theory: a cantilever of length L with a transverse
tip load P deflects at the free end by

    delta = P * L^3 / (3 * E * I)

where I is the second moment of area of the cross-section. For a rectangular
section of depth h (in-plane height) and thickness t (out-of-plane):

    I = t * h^3 / 12

Note: Euler-Bernoulli ignores shear deformation, so a 2D elasticity solution
converges to a value near (but not exactly equal to) this, especially for
slender beams. Our CST mesh will additionally under-predict due to the
element's bending stiffness (shear locking), approaching this value from
below as the mesh is refined.
"""


def cantilever_tip_deflection(P, L, E, h, t):
    """Euler-Bernoulli tip deflection of a cantilever under a tip load.

    Parameters
    ----------
    P : float
        Transverse tip load (magnitude).
    L : float
        Beam length.
    E : float
        Young's modulus.
    h : float
        Beam depth (in-plane height of the cross-section).
    t : float
        Beam thickness (out-of-plane).

    Returns
    -------
    delta : float
        Tip deflection magnitude.
    """
    I = t * h**3 / 12.0
    return P * L**3 / (3.0 * E * I)