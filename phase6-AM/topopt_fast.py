"""
SIMP topology optimization -- sparse, vectorized scaffolding for larger meshes.
Physics identical to topopt.py; assembly and filtering rewritten for speed so
180x60 runs in ~40s instead of needing ~4 GB of dense matrix.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def element_stiffness_Q4(E=1.0, nu=0.3):
    """Standard 8x8 stiffness for a unit square Q4 plane-stress element."""
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                  -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    return E/(1-nu**2)*np.array([
        [k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],
        [k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
        [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],
        [k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
        [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
        [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
        [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],
        [k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])


def build_edof(nelx, nely):
    """edof[e, :] = the 8 global DOFs for element e (column-major element order)."""
    edof = np.zeros((nelx*nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            e = elx*nely + ely
            n1 = (nely+1)*elx + ely
            n2 = (nely+1)*(elx+1) + ely
            edof[e] = [2*n1, 2*n1+1, 2*n2, 2*n2+1,
                       2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3]
    return edof


def build_filter(nelx, nely, rmin):
    """Precompute the sparse filter matrix H and its row sums.

    Computing the neighborhood weights once (instead of every iteration) turns
    the filter into a single sparse matrix-vector product per step.
    """
    iH, jH, sH = [], [], []
    r = int(np.ceil(rmin)-1)
    for i in range(nelx):
        for j in range(nely):
            e1 = i*nely + j
            for k in range(max(i-r,0), min(i+r+1, nelx)):
                for l in range(max(j-r,0), min(j+r+1, nely)):
                    e2 = k*nely + l
                    fac = rmin - np.sqrt((i-k)**2 + (j-l)**2)
                    if fac > 0:
                        iH.append(e1); jH.append(e2); sH.append(fac)
    H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsr()
    Hs = np.array(H.sum(axis=1)).flatten()
    return H, Hs


def fea_solve(nelx, nely, rho, penal, KE, edof, fixed, F):
    """Assemble K = sum rho^p KE as a sparse matrix, apply BCs, solve K u = F."""
    ndof = 2*(nelx+1)*(nely+1)
    x = rho.flatten(order='F')          # column-major, to match edof ordering
    sK = ((KE.flatten()[np.newaxis]).T * (x**penal)).flatten(order='F')
    iK = np.kron(edof, np.ones((8,1))).flatten()
    jK = np.kron(edof, np.ones((1,8))).flatten()
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    K = (K + K.T)/2.0
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = spsolve(K[free,:][:,free], F[free])
    return u


def compliance_and_sensitivity(nelx, nely, rho, penal, KE, edof, u):
    """c = sum rho^p ue' KE ue ;  dc/drho = -p rho^(p-1) ue' KE ue
    (the Phase 5 sensitivity, computed for all elements at once)."""
    x = rho.flatten(order='F')
    ce = np.einsum('ij,jk,ik->i', u[edof], KE, u[edof])
    c = np.sum((x**penal)*ce)
    dc = (-penal*(x**(penal-1))*ce).reshape((nely, nelx), order='F')
    return c, dc

def compliance_and_sensitivity_multi(nelx, nely, rho, penal, KE, edof, us,
                                     weights=None):
    """Weighted multi-load-case compliance and sensitivity.

    c = sum_k w_k * c_k,   so   dc/drho_j = -p rho_j^(p-1) * sum_k w_k * (ue_k' KE ue_k)

    The -p rho^(p-1) factor is the same for every load case (they share the
    same densities), so it factors out of the sum: accumulate strain energy
    across load cases, then multiply once.

    Parameters
    ----------
    us : list of displacement vectors, one per load case
    weights : list of floats, one per load case (defaults to all 1.0)
    """
    if weights is None:
        weights = np.ones(len(us))
    x = rho.flatten(order='F')
    ce_total = np.zeros(len(x))
    c = 0.0
    for u, w in zip(us, weights):
        ce = np.einsum('ij,jk,ik->i', u[edof], KE, u[edof])
        c += w*np.sum((x**penal)*ce)
        ce_total += w*ce
    dc = (-penal*(x**(penal-1))*ce_total).reshape((nely, nelx), order='F')
    return c, dc


def apply_filter(H, Hs, rho, dc):
    """Sensitivity filter as a single sparse matvec."""
    x = rho.flatten(order='F'); d = dc.flatten(order='F')
    dn = H.dot(x*d)/Hs/np.maximum(1e-3, x)
    return dn.reshape(dc.shape, order='F')


def mbb_boundary_conditions(nelx, nely):
    """MBB beam: symmetry on left edge (x fixed), roller at bottom-right
    (y fixed), unit downward load at top-left node."""
    ndof = 2*(nelx+1)*(nely+1)
    F = np.zeros(ndof)
    F[1] = -1.0
    fixed = np.union1d(np.arange(0, 2*(nely+1), 2), np.array([ndof-1]))
    return F, fixed