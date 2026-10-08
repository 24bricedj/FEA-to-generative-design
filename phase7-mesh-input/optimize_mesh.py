"""
SIMP topology optimization on ANY 2D triangle mesh.

Usage:
    python mesh_mbb.py 0.05          # make an unstructured MBB mesh -> mbb.msh
    python optimize_mesh.py mbb.msh  # optimize it

Boundary conditions come from the mesh's physical groups (by name), so the
solver never needs to know the geometry:
    "symmetry" -> x fixed      "roller" -> y fixed      "load" -> -1 in y
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from mesh_io import read_mesh
from fem_mesh import (precompute, build_filter, apply_filter,
                      fea_solve, compliance_and_sensitivity)


def oc_update(rho, V, volfrac, dc, move):

    """Phase 5 OC update, with the move limit passed in."""
    l1, l2 = 0.0, 1e9
    while (l2 - l1)/(l1 + l2 + 1e-30) > 1e-4:
        lmid = 0.5*(l1 + l2)
        B = np.maximum(-dc/(lmid*V), 1e-12)
        rho_new = np.maximum(0.001,
                    np.maximum(rho - move,
                        np.minimum(1.0,
                        np.minimum(rho + move,
                        rho*np.sqrt(B)))))
        if (rho_new*V).sum() > volfrac*V.sum():
            l1 = lmid
        else:
            l2 = lmid
    return rho_new


def move_schedule(it):
    if it < 60:  return 0.20
    if it < 150: return 0.10
    return 0.05


def main(fname, volfrac=0.5, penal=3.0, rmin=0.075, nit=300):
    nodes, tris, sets = read_mesh(fname)
    KE, V, edof, cent = precompute(nodes, tris)
    ndof = 2*len(nodes)
    print(f"mesh: {len(nodes)} nodes, {len(tris)} triangles, "
          f"element area range {V.min():.2e} .. {V.max():.2e}")

    # boundary conditions from physical groups
    fixed = np.union1d(2*sets["symmetry"], 2*sets["roller"] + 1)
    F = np.zeros(ndof)
    F[2*sets["load"] + 1] = -1.0

    H, Hs = build_filter(cent, rmin)
    rho = volfrac*np.ones(len(tris))
    c_hist = []

    for it in range(nit):
        u = fea_solve(rho, penal, KE, edof, ndof, fixed, F)
        c, dc = compliance_and_sensitivity(rho, penal, KE, edof, u)
        dc = apply_filter(H, Hs, rho, dc, V)
        rho = oc_update(rho, V, volfrac, dc, move_schedule(it))

        c_hist.append(c)
        if it % 25 == 0:
            vol = (rho*V).sum()/V.sum()
            print(f"  it {it:3d}  c={c:8.2f}  vol={vol:.3f}")
        if it > 80 and (max(c_hist[-20:]) - min(c_hist[-20:]))/c < 1e-3:
            print(f"  converged at it={it}")
            break

    vol = (rho*V).sum()/V.sum()
    grey = np.sum(V*4*rho*(1 - rho))/V.sum()
    print(f"final: c={c:.2f}  vol={vol:.3f}  greyness={grey:.3f}")

    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
    fig, ax = plt.subplots(figsize=(11, 4))
    tp = ax.tripcolor(tri, facecolors=rho, cmap="jet", vmin=0, vmax=1)
    ax.set_aspect("equal")
    ax.set_title(f"{fname}: {len(tris)} triangles, c = {c:.1f}")
    fig.colorbar(tp, ax=ax, label="density ρ")
    out = fname.rsplit(".", 1)[0] + "_result.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mbb.msh")