"""
Mesh-agnostic SIMP scaffolding on unstructured CST meshes.

Unlike the structured version, every element has its own shape, so:
  - each element's full-material stiffness is computed once and stored
  - element areas V vary and enter the volume constraint and filter
  - the filter finds neighbors by centroid distance (KD-tree), not grid index
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


def plane_stress_D(E, nu):
    return (E/(1-nu**2))*np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])


def element_stiffness(coords, E, nu, t):
    """Your Phase 2 CST element: k = B^T D B * t * A."""
    x1, y1 = coords[0]; x2, y2 = coords[1]; x3, y3 = coords[2]
    A = 0.5*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    b1, b2, b3 = y2-y3, y3-y1, y1-y2
    c1, c2, c3 = x3-x2, x1-x3, x2-x1
    B = (1/(2*A))*np.array([[b1, 0, b2, 0, b3, 0],
                            [0, c1, 0, c2, 0, c3],
                            [c1, b1, c2, b2, c3, b3]])
    return (B.T @ plane_stress_D(E, nu) @ B)*t*A


def precompute(nodes, tris, E=1.0, nu=0.3, t=1.0):
    """Per-element stiffness (ne,6,6), areas V (ne,), DOF map (ne,6),
    and centroids (ne,2). Done once: only rho changes between iterations."""
    ne = len(tris)
    KE = np.zeros((ne, 6, 6))
    for e in range(ne):
        KE[e] = element_stiffness(nodes[tris[e]], E, nu, t)
    x, y = nodes[:, 0], nodes[:, 1]
    a = tris
    V = 0.5*((x[a[:,1]]-x[a[:,0]])*(y[a[:,2]]-y[a[:,0]]) -
             (x[a[:,2]]-x[a[:,0]])*(y[a[:,1]]-y[a[:,0]]))
    edof = np.column_stack([2*a[:,0], 2*a[:,0]+1, 2*a[:,1], 2*a[:,1]+1,
                            2*a[:,2], 2*a[:,2]+1])
    cent = nodes[tris].mean(axis=1)
    return KE, V, edof, cent


def build_filter(cent, rmin):
    """Neighbors by centroid distance: weights w = rmin - dist (dist < rmin).
    rmin is now a PHYSICAL length, not a count of elements."""
    tree = cKDTree(cent)
    pairs = tree.query_pairs(rmin, output_type="ndarray")
    d = np.linalg.norm(cent[pairs[:, 0]] - cent[pairs[:, 1]], axis=1)
    w = rmin - d
    ne = len(cent)
    i = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(ne)])
    j = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(ne)])
    s = np.concatenate([w, w, np.full(ne, rmin)])     # self weight = rmin
    H = coo_matrix((s, (i, j)), shape=(ne, ne)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


def apply_filter(H, Hs, rho, dc, V):
    """Average sensitivity PER UNIT AREA over the neighborhood, then scale back
    to this element's size:  dc_j <- V_j/(rho_j Hs_j) * sum_k H_jk rho_k dc_k/V_k"""
    return V*H.dot(rho*dc/V)/(np.maximum(1e-3, rho)*Hs)


def fea_solve(rho, penal, KE, edof, ndof, fixed, F):
    sK = (KE*(rho**penal)[:, None, None]).ravel()
    iK = np.repeat(edof, 6, axis=1).ravel()
    jK = np.tile(edof, (1, 6)).ravel()
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = spsolve(K[free][:, free], F[free])
    return u


def compliance_and_sensitivity(rho, penal, KE, edof, u):
    ue = u[edof]
    ce = np.einsum('ei,eij,ej->e', ue, KE, ue)        # strain energy per element
    c = np.sum(rho**penal*ce)
    dc = -penal*rho**(penal-1)*ce
    return c, dc