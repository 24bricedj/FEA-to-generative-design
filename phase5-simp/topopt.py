"""
SIMP topology optimization -- MBB beam benchmark.

Scaffolding: mesh, FEA assembly/solve, sensitivity, filter, driver, plotting.
The OC update (the derived core) is written separately.
"""
import numpy as np
import matplotlib.pyplot as plt


def element_stiffness_Q4(E=1.0, nu=0.3):
    """Standard 8x8 stiffness for a unit square Q4 plane-stress element."""
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                  -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([
        [k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],
        [k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
        [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],
        [k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
        [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
        [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
        [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],
        [k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])
    return KE


def element_dofs_grid(elx, ely, nelx, nely):
    """Global DOF indices for element (elx, ely) in a nelx x nely grid."""
    n1 = (nely+1)*elx + ely
    n2 = (nely+1)*(elx+1) + ely
    return np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1,
                     2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])


def fea_solve(nelx, nely, rho, penal, KE, fixed, F):
    """Assemble K = sum rho^p KE, apply BCs, solve K u = F."""
    ndof = 2*(nelx+1)*(nely+1)
    K = np.zeros((ndof, ndof))
    for elx in range(nelx):
        for ely in range(nely):
            edof = element_dofs_grid(elx, ely, nelx, nely)
            K[np.ix_(edof, edof)] += (rho[ely, elx]**penal) * KE
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    return u


def compliance_and_sensitivity(nelx, nely, rho, penal, KE, u):
    """c = sum rho^p ue' KE ue ;  dc/drho = -p rho^(p-1) ue' KE ue
    (your Phase 5 sensitivity, applied element by element)."""
    c = 0.0
    dc = np.zeros((nely, nelx))
    for elx in range(nelx):
        for ely in range(nely):
            edof = element_dofs_grid(elx, ely, nelx, nely)
            ue = u[edof]
            ke_energy = ue @ KE @ ue
            c += (rho[ely, elx]**penal) * ke_energy
            dc[ely, elx] = -penal * rho[ely, elx]**(penal-1) * ke_energy
    return c, dc


def sensitivity_filter(nelx, nely, rmin, rho, dc):
    """Mesh-independence filter: weighted neighborhood average of dc,
    with linear hat weights (rmin - distance)."""
    dcn = np.zeros((nely, nelx))
    r = int(np.floor(rmin))
    for i in range(nelx):
        for j in range(nely):
            summ = 0.0
            for k in range(max(i-r, 0), min(i+r+1, nelx)):
                for l in range(max(j-r, 0), min(j+r+1, nely)):
                    fac = rmin - np.sqrt((i-k)**2 + (j-l)**2)
                    if fac > 0:
                        summ += fac
                        dcn[j, i] += fac * rho[l, k] * dc[l, k]
            dcn[j, i] /= (rho[j, i] * summ)
    return dcn


def mbb_boundary_conditions(nelx, nely):
    """MBB beam: symmetry on left edge (x fixed), roller at bottom-right
    (y fixed), unit downward load at top-left corner."""
    ndof = 2*(nelx+1)*(nely+1)
    F = np.zeros(ndof)
    F[1] = -1.0                       # downward load, top-left node
    fixed = np.union1d(
        np.arange(0, 2*(nely+1), 2),  # left edge: fix x (symmetry)
        np.array([ndof-1])            # bottom-right: fix y (roller)
    )
    return F, fixed


def plot_design(rho, title, savepath):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(-rho, cmap="gray", interpolation="nearest", vmin=-1, vmax=0)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_full_mbb(rho, title, savepath):
    """Mirror the half-model about its symmetry plane (the left edge) to
    show the complete MBB beam."""
    full = np.concatenate([np.fliplr(rho), rho], axis=1)
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.imshow(-full, cmap="gray", interpolation="nearest", vmin=-1, vmax=0)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)