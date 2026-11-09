"""3D assembly, solve, and loads. Imports YOUR element from fem3d.py."""
import numpy as np
from fem3d import element_stiffness_tet, tet_gradients


# ---------------------------------------------------------------------------
# 3D solver scaffolding: wiring your tet element into the optimizer
# ---------------------------------------------------------------------------
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def precompute_3d(nodes, tets, E=1.0, nu=0.3):
    """Per-element 12x12 stiffness, volumes, 12-DOF maps, and centroids."""
    ne = len(tets)
    KE = np.zeros((ne, 12, 12))
    V = np.zeros(ne)
    for e in range(ne):
        KE[e] = element_stiffness_tet(nodes[tets[e]], E, nu)
        V[e], _ = tet_gradients(nodes[tets[e]])
    edof = np.repeat(3*tets, 3, axis=1) + np.tile([0, 1, 2], 4)
    cent = nodes[tets].mean(axis=1)
    return KE, V, edof, cent


def fea_solve_3d(rho, penal, KE, edof, ndof, fixed, F):
    sK = (KE*(rho**penal)[:, None, None]).ravel()
    iK = np.repeat(edof, 12, axis=1).ravel()
    jK = np.tile(edof, (1, 12)).ravel()
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = spsolve(K[free][:, free], F[free])
    return u


def line_load(nodes, segments, ndof, direction, total):
    """Spread a total force evenly per unit length along tagged edge segments.
    Each segment hands half its share to each end node (consistent load for
    linear edges). direction: 0=x, 1=y, 2=z."""
    F = np.zeros(ndof)
    L = np.linalg.norm(nodes[segments[:, 1]] - nodes[segments[:, 0]], axis=1)
    q = total/L.sum()
    for (n1, n2), l in zip(segments, L):
        F[3*n1 + direction] += q*l/2
        F[3*n2 + direction] += q*l/2
    return F


def surface_load(nodes, faces, ndof, direction, total):

    F = np.zeros(ndof)
    a = nodes[faces[:, 0]]     # first corner of every triangle,  (k, 3)
    b = nodes[faces[:, 1]]     # second corner
    c = nodes[faces[:, 2]]     # third corner
    A = 0.5*np.linalg.norm(np.cross(b - a, c - a), axis=1)
    p = total/A.sum()
    for (n1, n2, n3), Ai in zip(faces, A):
        F[3*n1 + direction] += p*Ai/3
        F[3*n2 + direction] += p*Ai/3
        F[3*n3 + direction] += p*Ai/3
    
    return F

from scipy.sparse.linalg import splu


def fea_solve_3d_multi(rho, penal, KE, edof, ndof, fixed, Fs):
    """Solve K u = F for SEVERAL load cases at once.

    Fs : (ndof, n_cases) -- one column per load case
    Returns U : (ndof, n_cases) -- one displacement column per case

    K depends only on rho, so it's the same for every case: factorize it
    ONCE (the expensive part), then back-substitute each column (cheap).
    """
    sK = (KE*(rho**penal)[:, None, None]).ravel()
    iK = np.repeat(edof, 12, axis=1).ravel()
    jK = np.tile(edof, (1, 12)).ravel()
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    free = np.setdiff1d(np.arange(ndof), fixed)
    lu = splu(K[free][:, free])          # factorize once
    U = np.zeros_like(Fs)
    U[free] = lu.solve(Fs[free])         # every load case in one call
    return U