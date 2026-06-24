"""
2D steady-state heat conduction FEA solver (CST mesh).

Phase 3, step 2. By the same scalar-field simplification as 1D thermal, this
reuses the Phase 2 CST machinery but with temperature (1 DOF/node) instead of
displacement (2 DOF/node):

    - element matrix is 3×3 (not 6×6)
    - B is the 2×3 temperature-gradient matrix (not 3×6 strain-displacement)
    - constitutive is scalar k (not the 3×3 D matrix)

    k_e = Bᵀ B · k · A     where B = (1/2A) [[b1,b2,b3],[c1,c2,c3]]

Pipeline mirrors Phase 2 with 1-DOF-per-node assembly (like Phase 1).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Element conductivity matrix (YOUR JOB)
# ---------------------------------------------------------------------------

def element_conductivity_2d(coords, k):
    """Return the 3×3 element conductivity matrix for a CST element.

    k_e = Bᵀ B · k · A

    where B = (1/2A) [[b1, b2, b3], [c1, c2, c3]]
    is the 2×3 temperature-gradient matrix, and the b_i, c_i are the same
    shape-function gradient coefficients from Phase 2.

    Parameters
    ----------
    coords : ndarray of shape (3, 2)
        (x, y) coordinates of the triangle's three nodes.
    k : float
        Thermal conductivity.

    Returns
    -------
    ke : ndarray of shape (3, 3)
    """
    # TODO (user): transcribe — this is Phase 2's element_stiffness, simplified.
    #   1. unpack the 3 node coords (x1,y1,...,x3,y3)
    #   2. compute detC and A  (same as Phase 2)
    #   3. compute b1,b2,b3 and c1,c2,c3  (same as Phase 2)
    #   4. build the 2×3 B matrix
    #   5. return B.T @ B * k * A

    x1, y1 = coords[0,0] , coords[0,1]
    x2, y2 = coords[1,0] , coords[1,1]
    x3, y3 = coords[2,0] , coords[2,1]

    detC = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    A = detC / 2.0

    b1 = y2-y3
    b2 = y3-y1
    b3 = y1-y2
    c1 = x3-x2
    c2 = x1-x3
    c3 = x2-x1

    B = (1 / (2.0 * A)) * np.array([[b1, b2, b3] , [c1, c2, c3]])

    return B.T @ B * k *A


# ---------------------------------------------------------------------------
# Mesh (reused from Phase 2 — identical)
# ---------------------------------------------------------------------------

def build_mesh(width, height, nx, ny):
    """Structured triangular mesh. Identical to Phase 2."""
    xs = np.linspace(0.0, width, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([xs[i], ys[j]])
    nodes = np.array(nodes)

    def node_id(i, j):
        return j * (nx + 1) + i

    connectivity = []
    for j in range(ny):
        for i in range(nx):
            n00 = node_id(i, j)
            n10 = node_id(i + 1, j)
            n01 = node_id(i, j + 1)
            n11 = node_id(i + 1, j + 1)
            connectivity.append([n00, n10, n11])
            connectivity.append([n00, n11, n01])
    connectivity = np.array(connectivity)
    return nodes, connectivity


# ---------------------------------------------------------------------------
# Global assembly (1 DOF/node — like Phase 1, not Phase 2)
# ---------------------------------------------------------------------------

def assemble_conductivity(nodes, connectivity, k):
    """Assemble the global conductivity matrix. 1 DOF per node (scalar T),
    so the scatter is by node index directly — simpler than Phase 2's
    2-DOF version."""
    n_nodes = len(nodes)
    K = np.zeros((n_nodes, n_nodes))
    for element in connectivity:
        A_, B_, C_ = element[0], element[1], element[2]
        coords = nodes[[A_, B_, C_]]
        ke = element_conductivity_2d(coords, k)
        dofs = [A_, B_, C_]            # 1 DOF/node: dof index = node index
        K[np.ix_(dofs, dofs)] += ke
    return K


# ---------------------------------------------------------------------------
# Load, BCs, solve (reuse the 1D thermal versions — nonzero Dirichlet capable)
# ---------------------------------------------------------------------------

def apply_dirichlet_bc(K, F, fixed):
    """Nonzero-capable Dirichlet BCs — identical to the 1D thermal version.
    `fixed` is a dict {node_index: prescribed_temperature}."""
    n_nodes = K.shape[0]
    fixed_nodes = list(fixed.keys())
    all_nodes = np.arange(n_nodes)
    free_nodes = np.setdiff1d(all_nodes, fixed_nodes)

    F_mod = F.copy()
    for j, Tbar in fixed.items():
        F_mod -= K[:, j] * Tbar

    K_red = K[np.ix_(free_nodes, free_nodes)]
    F_red = F_mod[free_nodes]
    return K_red, F_red, free_nodes


def solve_system(K_red, F_red):
    return np.linalg.solve(K_red, F_red)


def reconstruct_full_solution(T_red, free_nodes, n_nodes, fixed):
    T_full = np.zeros(n_nodes)
    T_full[free_nodes] = T_red
    for j, Tbar in fixed.items():
        T_full[j] = Tbar
    return T_full