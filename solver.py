"""
1D linear FEA solver for the bar problem.

This module implements the finite element method derived from first principles
in our Phase 1 notes. The solver follows the standard pipeline:

    mesh -> element stiffness -> global assembly -> load vector
                                    -> apply BCs -> solve -> return u

The physics core functions are implemented here. The driver script
(run_example.py) sets up the problem parameters and orchestrates the calls.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Mesh construction (scaffolding — provided)
# ---------------------------------------------------------------------------

def build_mesh(L, n_elements):
    """Build a uniform 1D mesh.

    Parameters
    ----------
    L : float
        Total length of the bar.
    n_elements : int
        Number of elements (equal length).

    Returns
    -------
    nodes : ndarray of shape (n_nodes,)
        Positions of the nodes. n_nodes = n_elements + 1.
    connectivity : ndarray of shape (n_elements, 2)
        For each element, the global indices of its two nodes (local 0, local 1).
    """
    n_nodes = n_elements + 1
    nodes = np.linspace(0.0, L, n_nodes)
    connectivity = np.array([[i, i + 1] for i in range(n_elements)])
    return nodes, connectivity


# ---------------------------------------------------------------------------
# Element stiffness (YOUR JOB)
# ---------------------------------------------------------------------------

def element_stiffness(E, A, h_e):
    """Return the 2x2 element stiffness matrix for a 1D linear bar element.

    From the hand derivation:
        k_e = (EA / h_e) * [[ 1, -1],
                            [-1,  1]]

    Parameters
    ----------
    E : float
        Young's modulus.
    A : float
        Cross-sectional area.
    h_e : float
        Element length.

    Returns
    -------
    k_e : ndarray of shape (2, 2)
        Element stiffness matrix.
    """
    
    k_e = (E*A / h_e) * np.array([[1,-1], [-1,1]])

    return k_e


# ---------------------------------------------------------------------------
# Global assembly of K (YOUR JOB)
# ---------------------------------------------------------------------------

def assemble_stiffness(nodes, connectivity, E, A):
    """Assemble the global stiffness matrix K by scattering element matrices.

    Parameters
    ----------
    nodes : ndarray of shape (n_nodes,)
        Node positions.
    connectivity : ndarray of shape (n_elements, 2)
        Element-to-global node mapping.
    E, A : float
        Material and section properties (uniform).

    Returns
    -------
    K : ndarray of shape (n_nodes, n_nodes)
        Global stiffness matrix.
    """
    n_nodes = len(nodes)
    K = np.zeros((n_nodes, n_nodes))

    for I, J in connectivity:
        h_e = nodes[J]-nodes[I]             # length of this element
        k_e = element_stiffness(E, A, h_e)              # call element_stiffness
        K[I, I] += k_e[0, 0]
        K[I, J] += k_e[0, 1]
        K[J, I] += k_e[1, 0]
        K[J, J] += k_e[1, 1]
    return K




# ---------------------------------------------------------------------------
# Load vector (YOUR JOB)
# ---------------------------------------------------------------------------

def assemble_load(nodes, connectivity, b0, point_loads):
    """Assemble the global load vector F.

    For a uniform body load b0, each element contributes b0*h_e/2 to each
    of its two nodes (this is the consistent load for linear shape functions
    with a constant body force).

    Point loads add directly to the load vector entry for their node.

    Parameters
    ----------
    nodes : ndarray of shape (n_nodes,)
    connectivity : ndarray of shape (n_elements, 2)
    b0 : float
        Uniform body load (force per unit length).
    point_loads : list of (node_index, force) tuples
        Each tuple adds `force` to F[node_index].

    Returns
    -------
    F : ndarray of shape (n_nodes,)
        Global load vector.
    """
    n_nodes = len(nodes)
    F = np.zeros(n_nodes)

# body load contribution
    for I, J in connectivity:
        h_e = nodes[J]-nodes[I]
        F[I] += b0*h_e/2
        F[J] += b0*h_e/2

    # point load contribution
    for node_index, force in point_loads:
        F[node_index] += force

    return F



# ---------------------------------------------------------------------------
# Apply Dirichlet BC (YOUR JOB)
# ---------------------------------------------------------------------------

def apply_dirichlet_bc(K, F, fixed_nodes):
    """Apply essential (Dirichlet) BCs by direct elimination.

    For each fixed node, remove the corresponding row and column from K and
    the corresponding entry from F. We only handle u = 0 BCs here (not
    nonzero prescribed displacements), which is enough for our problem.

    Parameters
    ----------
    K : ndarray, square
    F : ndarray, 1D
    fixed_nodes : list of int
        Global indices of nodes where u is prescribed to be zero.

    Returns
    -------
    K_red : ndarray
        Reduced stiffness matrix.
    F_red : ndarray
        Reduced load vector.
    free_nodes : ndarray
        Indices of the unconstrained nodes (used later to reconstruct full u).
    """
    n_nodes = K.shape[0]
    all_nodes = np.arange(n_nodes)
    free_nodes = np.setdiff1d(all_nodes, fixed_nodes)

    K_red = K[np.ix_(free_nodes, free_nodes)]
    F_red = F[free_nodes]
    return K_red, F_red, free_nodes


# ---------------------------------------------------------------------------
# Solve and reconstruct (scaffolding — provided)
# ---------------------------------------------------------------------------

def solve_system(K_red, F_red):
    """Solve the reduced linear system K_red u_red = F_red."""
    return np.linalg.solve(K_red, F_red)


def reconstruct_full_solution(u_red, free_nodes, n_nodes):
    """Place the reduced solution back into the full nodal displacement vector,
    with zeros at the constrained nodes."""
    u_full = np.zeros(n_nodes)
    u_full[free_nodes] = u_red
    return u_full

def evaluate_fea_solution(x_query, nodes, u_nodal):
    """Evaluate the piecewise-linear FEA solution at arbitrary query points.

    Linear shape functions mean the FEA solution between nodes is just a
    straight line connecting the nodal values. NumPy's np.interp does
    exactly this kind of linear interpolation.

    Parameters
    ----------
    x_query : ndarray
        Points at which to evaluate the FEA solution.
    nodes : ndarray
        Node positions (must be sorted ascending).
    u_nodal : ndarray
        FEA displacement at each node.

    Returns
    -------
    u_query : ndarray
        FEA displacement at each query point.
    """
    return np.interp(x_query, nodes, u_nodal)