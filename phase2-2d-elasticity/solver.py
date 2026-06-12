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

# ---------------------------------------------------------------------------
# Mesh construction (scaffolding — provided)
# ---------------------------------------------------------------------------

def build_mesh(width, height, nx, ny):
    """Build a structured triangular mesh of a rectangular domain.

    The rectangle [0, width] x [0, height] is divided into an nx-by-ny grid
    of cells; each cell is split into two CST triangles.

    Parameters
    ----------
    width, height : float
        Dimensions of the rectangular domain.
    nx, ny : int
        Number of cells in the x and y directions.

    Returns
    -------
    nodes : ndarray of shape (n_nodes, 2)
        (x, y) coordinates of each node. n_nodes = (nx+1) * (ny+1).
    connectivity : ndarray of shape (n_elements, 3)
        For each triangle, the global indices of its 3 nodes (CCW order).
        n_elements = 2 * nx * ny.
    """
    # --- nodes: a regular grid of (nx+1) x (ny+1) points ---
    xs = np.linspace(0.0, width, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)

    nodes = []
    for j in range(ny + 1):       # row index (y)
        for i in range(nx + 1):   # column index (x)
            nodes.append([xs[i], ys[j]])
    nodes = np.array(nodes)

    # helper: global node index from grid position (i, j)
    def node_id(i, j):
        return j * (nx + 1) + i

    # --- elements: split each grid cell into two triangles ---
    connectivity = []
    for j in range(ny):
        for i in range(nx):
            # corners of this cell
            n00 = node_id(i,     j)      # bottom-left
            n10 = node_id(i + 1, j)      # bottom-right
            n01 = node_id(i,     j + 1)  # top-left
            n11 = node_id(i + 1, j + 1)  # top-right

            # two triangles, both counterclockwise
            connectivity.append([n00, n10, n11])   # lower-right triangle
            connectivity.append([n00, n11, n01])   # upper-left triangle
    connectivity = np.array(connectivity)

    return nodes, connectivity

def assemble_stiffness(nodes, connectivity, E, nu, t):
    # 1. figure out total number of DOFs (2 per node)
    # 2. make an empty global K of that size
    # 3. for each element:
    #      a. get its 3 node indices from connectivity
    #      b. look up their coordinates from nodes
    #      c. compute the 6x6 element stiffness
    #      d. build the list of 6 global DOF indices
    #      e. scatter the 6x6 into K at those DOFs
    # 4. return K

    n_nodes = len(nodes)
    n_dofs = 2 * n_nodes
    K = np.zeros((n_dofs, n_dofs))

    for element in connectivity:
        A, B, C = element[0], element[1], element[2]

        coords = nodes[[A, B, C]]
        
        k = element_stiffness(coords, E, nu, t)

        dofs = [2*A, 2*A+1, 2*B, 2*B+1, 2*C, 2*C+1]
        
        K[np.ix_(dofs, dofs)] += k
        
    return K


def apply_dirichlet_bc(K, F, fixed_dofs):
    """Apply zero-displacement Dirichlet BCs by direct elimination.

    Parameters
    ----------
    K : ndarray, (n_dofs, n_dofs)
    F : ndarray, (n_dofs,)
    fixed_dofs : list of int
        Global DOF indices to constrain to zero. Note these are DOF indices,
        not node indices: node n's DOFs are 2n (x) and 2n+1 (y).

    Returns
    -------
    K_red, F_red, free_dofs
    """
    n_dofs = K.shape[0]
    all_dofs = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    K_red = K[np.ix_(free_dofs, free_dofs)]
    F_red = F[free_dofs]

    return K_red, F_red, free_dofs

def assemble_load(n_dofs, point_loads):
    """Assemble the global load vector from point loads.

    Parameters
    ----------
    n_dofs : int
        Total number of DOFs (2 * n_nodes).
    point_loads : list of (dof_index, force) tuples
        Each adds `force` to F[dof_index]. Note dof_index, not node index:
        node n's y-DOF (vertical) is 2n+1, x-DOF (horizontal) is 2n.

    Returns
    -------
    F : ndarray of shape (n_dofs,)
    """
    F = np.zeros(n_dofs)
    for dof_index, force in point_loads:
        F[dof_index] += force
    return F

def solve_system(K_red, F_red):
    """Solve the reduced linear system K_red u_red = F_red."""
    return np.linalg.solve(K_red, F_red)


def reconstruct_full_solution(u_red, free_dofs, n_dofs):
    """Place the reduced solution back into the full DOF vector, zeros at fixed DOFs."""
    u_full = np.zeros(n_dofs)
    u_full[free_dofs] = u_red
    return u_full