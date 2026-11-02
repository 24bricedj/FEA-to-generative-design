"""
3D linear tetrahedron (4 nodes, 3 DOF/node, 12 DOF element).

The 3D analog of the Phase 2 CST: linear shape functions -> constant strain,
so the element integral collapses to  k = B^T D B * V.
"""
import numpy as np


def iso_D_3d(E, nu):
    """6x6 isotropic constitutive matrix.
    Strain order: [eps_x, eps_y, eps_z, gamma_xy, gamma_yz, gamma_xz]."""
    c = E/((1 + nu)*(1 - 2*nu))
    D = np.zeros((6, 6))
    D[:3, :3] = nu
    np.fill_diagonal(D[:3, :3], 1 - nu)
    D[3:, 3:] = np.eye(3)*(1 - 2*nu)/2
    return c*D


def tet_gradients(coords):
    """Volume V and shape-function gradients for a tet.

    coords : (4, 3) node coordinates

    The shape functions N_i = a_i + b_i x + c_i y + d_i z must equal 1 at node i
    and 0 at the other three. Stacking those conditions gives  C @ coeffs = I,
    where C is the 4x4 matrix [1 x y z] you wrote by hand. So the coefficients
    are simply the columns of inv(C) -- and det(C) = 6V.

    Returns
    -------
    V : float
    G : (3, 4) array.  G[:, i] = [dN_i/dx, dN_i/dy, dN_i/dz]
        (these already include the 1/(6V) factor: they are b_i/6V, c_i/6V, d_i/6V)
    """
    C = np.hstack([np.ones((4, 1)), coords])
    V = np.linalg.det(C)/6.0
    G = np.linalg.inv(C)[1:, :]
    return V, G


def element_stiffness_tet(coords, E, nu):
    # 1. volume and gradients
    V, G = tet_gradients(coords)

    # 2. empty B, then fill one node's block at a time
    B = np.zeros((6, 12))
    for i in range(4):
        bx, by, bz = G[:, i]
        B[:, 3*i:3*i+3] = np.array([
            [bx, 0,  0 ],
            [0, by, 0],           # (A) eps_y row
            [0, 0, bz],           # (B) eps_z row
            [by, bx, 0 ],
            [0, bz, by],           # (C) gamma_yz row
            [bz, 0, bx],           # (D) gamma_xz row
        ])

    # 3. stiffness
    D = iso_D_3d(E, nu)
    return B.T @ D @ B * V
    raise NotImplementedError("Build B and return the tet stiffness.")