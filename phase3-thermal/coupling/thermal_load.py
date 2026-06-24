import numpy as np


def plane_stress_D(E, nu):
    """Plane-stress constitutive matrix D (3×3). Same as Phase 2."""
    return (E / (1.0 - nu**2)) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])



def element_B_matrix(coords):
    """Compute the 3×6 strain-displacement matrix B for a CST element.
    (Same B as Phase 2's element_stiffness — pulled out so the thermal
    load can reuse it.)"""
    x1, y1 = coords[0, 0], coords[0, 1]
    x2, y2 = coords[1, 0], coords[1, 1]
    x3, y3 = coords[2, 0], coords[2, 1]
    detC = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    A = detC / 2.0
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
    B = (1.0 / (2.0 * A)) * np.array([
        [b1, 0, b2, 0, b3, 0],
        [0, c1, 0, c2, 0, c3],
        [c1, b1, c2, b2, c3, b3],
    ])
    return B, A


def assemble_thermal_load(nodes, connectivity, T, T_ref, E, nu, alpha, t):
    """Assemble the global thermal load vector F_th.

    Parameters
    ----------
    nodes : (n_nodes, 2)
    connectivity : (n_elements, 3)
    T : (n_nodes,)
        Nodal temperature field (from the thermal solver).
    T_ref : float
        Stress-free reference temperature.
    E, nu : float
        Elastic properties (for D).
    alpha : float
        Coefficient of thermal expansion.
    t : float
        Thickness.

    Returns
    -------
    F_th : (2 * n_nodes,)
    """
    n_nodes = len(nodes)
    n_dofs = 2 * n_nodes
    F_th = np.zeros(n_dofs)

    D = plane_stress_D(E, nu)       # 3×3, same as Phase 2

    for element in connectivity:
        A_, B_, C_ = element[0], element[1], element[2]
        coords = nodes[[A_, B_, C_]]
        Bmat, area = element_B_matrix(coords)

        dT = (T[A_] + T[B_] + T[C_]) / 3.0 - T_ref
        eps_th = np.array([alpha * dT, alpha * dT, 0.0])
        f_th = Bmat.T @ D @ eps_th * t * area
        dofs = [2*A_ , 2*A_+1, 2*B_ , 2*B_+1, 2*C_ , 2*C_+1]
        F_th [dofs] += f_th

    return F_th