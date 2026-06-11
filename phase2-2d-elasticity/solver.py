"""
2D linear elastic FEA solver (plane stress, constant-strain triangle).

Phase 2 of the FEA-to-generative-design project. Generalizes the Phase 1
1D bar solver to 2D elasticity using CST elements:

    - displacement is now a vector (u, v): 2 DOFs per node
    - elements are triangles (3 nodes, 6 DOFs)
    - element stiffness is k = Bᵀ D B · t · A   (6×6)

Pipeline mirrors Phase 1:
    mesh -> element stiffness -> global assembly -> load vector
                                    -> apply BCs -> solve -> post-process
"""

import numpy as np


# ---------------------------------------------------------------------------
# Constitutive matrix (scaffolding — provided)
# ---------------------------------------------------------------------------

def plane_stress_D(E, nu):
    """Plane-stress constitutive matrix D (3×3).

    Relates stress [σx, σy, τxy]ᵀ = D · [εx, εy, γxy]ᵀ.

        D = E/(1-ν²) · [[1,  ν,        0      ],
                        [ν,  1,        0      ],
                        [0,  0,  (1-ν)/2     ]]

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    D : ndarray of shape (3, 3)
    """
    return (E / (1.0 - nu**2)) * np.array([
        [1.0, nu,           0.0],
        [nu,  1.0,          0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])


# ---------------------------------------------------------------------------
# Element stiffness (YOUR JOB)
# ---------------------------------------------------------------------------

def element_stiffness(coords, E, nu, t):
    """Return the 6×6 element stiffness matrix for a CST element.

    From the hand derivation:
        k = Bᵀ D B · t · A

    where
        - A is the triangle area,
        - D is the 3×3 plane-stress constitutive matrix,
        - B is the 3×6 strain-displacement matrix built from the shape-
          function gradients (b_i, c_i coefficients), with DOF ordering
          [u1, v1, u2, v2, u3, v3].

    Parameters
    ----------
    coords : ndarray of shape (3, 2)
        The (x, y) coordinates of the triangle's three nodes, row i = node i.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    t : float
        Element thickness (plane stress).

    Returns
    -------
    k : ndarray of shape (6, 6)
        Element stiffness matrix, DOF order [u1, v1, u2, v2, u3, v3].
    """


    x1, y1 = coords[0, 0], coords[0, 1]   # node 1's x and y
    x2, y2 = coords[1, 0], coords[1, 1]   # node 2's x and y
    x3, y3 = coords[2, 0], coords[2, 1]   # node 3's x and y


    detC = x1 * (y2-y3) + x2 * (y3-y1) + x3 * (y1-y2)
    A = detC/2.0

    b1 = y2-y3
    b2 = y3-y1
    b3 = y1-y2

    c1 = x3-x2
    c2 = x1-x3
    c3 = x2-x1

    B = (1.0/(2.0*A)) * np.array([[b1, 0, b2, 0, b3, 0], [0, c1, 0, c2, 0, c3], [c1, b1, c2, b2, c3, b3]])

    D = plane_stress_D(E, nu)

    k = (B.T @ D @ B) * t * A

    return k