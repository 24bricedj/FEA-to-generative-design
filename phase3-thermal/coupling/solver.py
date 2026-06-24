"""
Elasticity solver pieces for the coupled thermal-structural solve.
Self-contained copy of the Phase 2 machinery plus stress recovery with
the thermal correction.
"""
import numpy as np
from thermal_load import plane_stress_D, element_B_matrix


def build_mesh(width, height, nx, ny):
    xs = np.linspace(0.0, width, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([xs[i], ys[j]])
    nodes = np.array(nodes)
    def node_id(i, j): return j * (nx + 1) + i
    conn = []
    for j in range(ny):
        for i in range(nx):
            n00 = node_id(i, j); n10 = node_id(i+1, j)
            n01 = node_id(i, j+1); n11 = node_id(i+1, j+1)
            conn.append([n00, n10, n11]); conn.append([n00, n11, n01])
    return nodes, np.array(conn)


def element_stiffness(coords, E, nu, t):
    B, A = element_B_matrix(coords)
    D = plane_stress_D(E, nu)
    return (B.T @ D @ B) * t * A


def assemble_stiffness(nodes, connectivity, E, nu, t):
    n = len(nodes); ndof = 2*n; K = np.zeros((ndof, ndof))
    for el in connectivity:
        A_, B_, C_ = el[0], el[1], el[2]
        coords = nodes[[A_, B_, C_]]
        k = element_stiffness(coords, E, nu, t)
        dofs = [2*A_, 2*A_+1, 2*B_, 2*B_+1, 2*C_, 2*C_+1]
        K[np.ix_(dofs, dofs)] += k
    return K


def apply_dirichlet_bc(K, F, fixed_dofs):
    ndof = K.shape[0]
    free = np.setdiff1d(np.arange(ndof), fixed_dofs)
    return K[np.ix_(free, free)], F[free], free


def solve_system(Kr, Fr):
    return np.linalg.solve(Kr, Fr)


def reconstruct_full_solution(ur, free, ndof):
    u = np.zeros(ndof); u[free] = ur; return u


def recover_element_stress(coords, u_elem, E, nu, alpha, dT):
    """Stress in one element including the thermal correction:
       sigma = D (B u - eps_th)."""
    B, A = element_B_matrix(coords)
    D = plane_stress_D(E, nu)
    eps_total = B @ u_elem
    eps_th = np.array([alpha*dT, alpha*dT, 0.0])
    return D @ (eps_total - eps_th)