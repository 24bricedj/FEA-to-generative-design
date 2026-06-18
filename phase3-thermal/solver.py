"""
1D steady-state heat conduction FEA solver.

Phase 3 of the FEA-to-generative-design project. By the isomorphism with
Phase 1 elasticity:

    u (displacement)  ->  T (temperature)
    E (Young's mod.)  ->  k (thermal conductivity)
    b (body load)     ->  Q (heat source)
    EA du/dx (force)  ->  kA dT/dx (heat flow rate, Fourier's law)

The governing equation is structurally identical to the Phase 1 bar:

    d/dx(kA dT/dx) + Q = 0

so the mesh, element matrix, assembly, and solve carry over directly. The one
new piece is NONZERO Dirichlet BCs (fixed end temperatures T0, TL != 0), which
Phase 1 did not need (it only fixed displacements to zero).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Mesh construction (carries over from Phase 1 — identical)
# ---------------------------------------------------------------------------

def build_mesh(L, n_elements):
    """Build a uniform 1D mesh. Identical to Phase 1."""
    n_nodes = n_elements + 1
    nodes = np.linspace(0.0, L, n_nodes)
    connectivity = np.array([[i, i + 1] for i in range(n_elements)])
    return nodes, connectivity


# ---------------------------------------------------------------------------
# Element conductivity matrix (isomorphism: E -> k)
# ---------------------------------------------------------------------------

def element_conductivity(k, A, h_e):
    """Return the 2x2 element conductivity matrix for a 1D bar element.

    By the isomorphism with Phase 1's element_stiffness (E -> k):
        k_e = (k * A / h_e) * [[ 1, -1], [-1,  1]]

    Parameters
    ----------
    k : float
        Thermal conductivity.
    A : float
        Cross-sectional area.
    h_e : float
        Element length.

    Returns
    -------
    k_e : ndarray of shape (2, 2)
    """
    return (k * A / h_e) * np.array([[1.0, -1.0], [-1.0, 1.0]])


# ---------------------------------------------------------------------------
# Global assembly (carries over from Phase 1 — identical scatter)
# ---------------------------------------------------------------------------

def assemble_conductivity(nodes, connectivity, k, A):
    """Assemble the global conductivity matrix K. Identical structure to
    Phase 1's assemble_stiffness, with E -> k."""
    n_nodes = len(nodes)
    K = np.zeros((n_nodes, n_nodes))
    for I, J in connectivity:
        h_e = nodes[J] - nodes[I]
        k_e = element_conductivity(k, A, h_e)
        K[I, I] += k_e[0, 0]
        K[I, J] += k_e[0, 1]
        K[J, I] += k_e[1, 0]
        K[J, J] += k_e[1, 1]
    return K


# ---------------------------------------------------------------------------
# Load vector (heat source Q — isomorphism: b -> Q)
# ---------------------------------------------------------------------------

def assemble_load(nodes, connectivity, Q0):
    """Assemble the global load vector from a uniform heat source Q0.

    By the isomorphism (b -> Q), each element contributes Q0*h_e/2 to each
    of its two nodes — identical consistent-load structure to Phase 1.
    (No point-load term here; boundary temperatures are handled in the BC step.)

    Parameters
    ----------
    nodes : ndarray
    connectivity : ndarray
    Q0 : float
        Uniform volumetric heat source.

    Returns
    -------
    F : ndarray of shape (n_nodes,)
    """
    n_nodes = len(nodes)
    F = np.zeros(n_nodes)
    for I, J in connectivity:
        h_e = nodes[J] - nodes[I]
        F[I] += Q0 * h_e / 2.0
        F[J] += Q0 * h_e / 2.0
    return F


# ---------------------------------------------------------------------------
# Apply Dirichlet BCs — NONZERO capable (YOUR JOB — the new piece)
# ---------------------------------------------------------------------------

def apply_dirichlet_bc(K, F, fixed):
    """Apply (possibly nonzero) Dirichlet BCs by elimination.

    Implements the method you derived: a prescribed value at node j injects
    a load -K[i,j]*Tbar_j into every free equation i, then row/col j are
    removed.

    Parameters
    ----------
    K : ndarray, (n_nodes, n_nodes)
    F : ndarray, (n_nodes,)
    fixed : dict {node_index: prescribed_temperature}
        e.g. {0: 100.0, n_last: 20.0} fixes node 0 to 100 and the last node
        to 20. Values may be nonzero.

    Returns
    -------
    K_red, F_red, free_nodes
    """
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



# ---------------------------------------------------------------------------
# Solve and reconstruct (carries over — but reconstruct must restore fixed vals)
# ---------------------------------------------------------------------------

def solve_system(K_red, F_red):
    """Solve the reduced linear system. Identical to Phase 1."""
    return np.linalg.solve(K_red, F_red)


def reconstruct_full_solution(T_red, free_nodes, n_nodes, fixed):
    """Place the reduced solution back into the full temperature vector,
    inserting the prescribed values at the fixed nodes.

    Unlike Phase 1 (which put zeros at fixed nodes), here the fixed nodes hold
    their prescribed NONZERO temperatures.

    Parameters
    ----------
    T_red : ndarray
        Solution at the free nodes.
    free_nodes : ndarray
    n_nodes : int
    fixed : dict {node_index: prescribed_temperature}

    Returns
    -------
    T_full : ndarray of shape (n_nodes,)
    """
    T_full = np.zeros(n_nodes)
    T_full[free_nodes] = T_red
    for j, Tbar in fixed.items():
        T_full[j] = Tbar
    return T_full