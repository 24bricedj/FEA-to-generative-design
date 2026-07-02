"""Phase 2 elasticity machinery, extended with per-element densities for
topology optimization. K = sum_i rho_i * K_i^(0)  (linear SIMP, p=1)."""
import numpy as np


def plane_stress_D(E, nu):
    return (E / (1.0 - nu**2)) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])


def element_stiffness(coords, E, nu, t):
    x1, y1 = coords[0]; x2, y2 = coords[1]; x3, y3 = coords[2]
    detC = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2); A = detC/2.0
    b1, b2, b3 = y2-y3, y3-y1, y1-y2
    c1, c2, c3 = x3-x2, x1-x3, x2-x1
    B = (1.0/(2.0*A)) * np.array([
        [b1, 0, b2, 0, b3, 0],
        [0, c1, 0, c2, 0, c3],
        [c1, b1, c2, b2, c3, b3],
    ])
    D = plane_stress_D(E, nu)
    return (B.T @ D @ B) * t * A


def build_mesh(width, height, nx, ny):
    xs = np.linspace(0, width, nx+1); ys = np.linspace(0, height, ny+1)
    nodes = []
    for j in range(ny+1):
        for i in range(nx+1):
            nodes.append([xs[i], ys[j]])
    nodes = np.array(nodes)
    def nid(i, j): return j*(nx+1)+i
    conn = []
    for j in range(ny):
        for i in range(nx):
            n00 = nid(i, j); n10 = nid(i+1, j)
            n01 = nid(i, j+1); n11 = nid(i+1, j+1)
            conn.append([n00, n10, n11]); conn.append([n00, n11, n01])
    return nodes, np.array(conn)


def element_dofs(el):
    """The 6 global DOF indices for a triangle's 3 nodes."""
    A_, B_, C_ = el
    return [2*A_, 2*A_+1, 2*B_, 2*B_+1, 2*C_, 2*C_+1]


def assemble_stiffness(nodes, conn, rho, E, nu, t):
    """Density-weighted global stiffness: K = sum_i rho_i * K_i^(0)."""
    n = len(nodes); ndof = 2*n; K = np.zeros((ndof, ndof))
    for ei, el in enumerate(conn):
        k0 = element_stiffness(nodes[el], E, nu, t)
        dofs = element_dofs(el)
        K[np.ix_(dofs, dofs)] += rho[ei] * k0
    return K


def apply_bc(K, F, fixed):
    ndof = K.shape[0]
    free = np.setdiff1d(np.arange(ndof), fixed)
    return K[np.ix_(free, free)], F[free], free


def solve(Kr, Fr):
    return np.linalg.solve(Kr, Fr)


def reconstruct(ur, free, ndof):
    u = np.zeros(ndof); u[free] = ur; return u