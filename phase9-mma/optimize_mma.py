"""
MMA vs OC on the 2D unstructured MBB beam.

  OC   + sensitivity filter   (your Phase 7 setup)
  MMA  + sensitivity filter   (same gradients, new optimizer)
  MMA  + density filter       (exact gradients: the setup stress constraints need)

    python ..\\phase7-mesh-input\\mesh_mbb.py 0.05      (if mbb.msh isn't here yet)
    python optimize_mma.py ..\\phase7-mesh-input\\mbb.msh
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "phase7-mesh-input"))
from mesh_io import read_mesh                                     # noqa: E402
from fem_mesh import (precompute, build_filter, apply_filter,     # noqa: E402
                      fea_solve, compliance_and_sensitivity)
from optimize_mesh import oc_update, move_schedule                # noqa: E402
from mma import MMA                                               # noqa: E402
from density_filter import DensityFilter                          # noqa: E402


def run(fname, method, volfrac=0.5, penal=3.0, rmin=0.075, nit=300):
    nodes, tris, sets = read_mesh(fname)
    KE, V, edof, cent = precompute(nodes, tris)
    ndof = 2*len(nodes)
    fixed = np.union1d(2*sets["symmetry"], 2*sets["roller"] + 1)
    F = np.zeros(ndof)
    F[2*sets["load"] + 1] = -1.0
    H, Hs = build_filter(cent, rmin)
    df = DensityFilter(cent, V, rmin) if method == "MMA+DF" else None

    rho = volfrac*np.ones(len(tris))
    opt = MMA(len(tris)) if method.startswith("MMA") else None
    hist = []
    t0 = time.time()
    for it in range(nit):
        if method == "MMA+DF":
            rho_t = df.forward(rho)                       # the physics sees rho_t
            u = fea_solve(rho_t, penal, KE, edof, ndof, fixed, F)
            c, dc_t = compliance_and_sensitivity(rho_t, penal, KE, edof, u)
            dc = df.backward(dc_t)                        # exact dc/d(rho)
            dv = df.backward(V)                           # exact d(volume)/d(rho)
            rho_new = opt.update(rho, dc, dv, volfrac*V.sum())
            # measure change on what the physics sees: a design-variable wiggle
            # that the filter averages away doesn't change the structure
            change = np.abs(df.forward(rho_new) - rho_t).max()
            rho = rho_new
            hist.append(c)
            if it % 10 == 0:
                print(f"  {method} it {it:3d}  c={c:8.3f}  vol={(df.forward(rho)*V).sum()/V.sum():.3f}  change={change:.3f}")
            if it >= 30 and change < 0.01:
                break
            continue
        u = fea_solve(rho, penal, KE, edof, ndof, fixed, F)
        c, dc = compliance_and_sensitivity(rho, penal, KE, edof, u)
        dc = apply_filter(H, Hs, rho, dc, V)
        if method == "MMA":
            rho_new = opt.update(rho, dc, V, volfrac*V.sum())
        else:
            rho_new = oc_update(rho, V, volfrac, dc, move_schedule(it))
        change = np.abs(rho_new - rho).max()
        rho = rho_new
        hist.append(c)
        if it % 10 == 0:
            print(f"  {method} it {it:3d}  c={c:8.3f}  vol={(rho*V).sum()/V.sum():.3f}  change={change:.3f}")
        if it >= 30 and change < 0.01:
            break
    if method == "MMA+DF":
        rho = df.forward(rho)          # report and plot what the physics saw
    grey = np.sum(V*4*rho*(1 - rho))/V.sum()
    print(f"{method}: c = {c:.2f} after {it+1} iterations, vol = {(rho*V).sum()/V.sum():.3f}, "
          f"greyness = {grey:.3f}  ({time.time()-t0:.0f}s)\n")
    return nodes, tris, rho, hist


def main(fname):
    methods = ["OC", "MMA", "MMA+DF"]
    results = {m: run(fname, m) for m in methods}
    fig, axes = plt.subplots(4, 1, figsize=(10, 13))
    for ax, m in zip(axes[:3], methods):
        nodes, tris, rho, hist = results[m]
        tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
        ax.tripcolor(tri, facecolors=rho, cmap="gray_r", vmin=0, vmax=1)
        ax.set_aspect("equal")
        ax.set_title(f"{m}: c = {hist[-1]:.2f} ({len(hist)} iterations)")
        ax.axis("off")
    for m in methods:
        axes[3].semilogy(results[m][3], label=m)
    axes[3].set_xlabel("iteration")
    axes[3].set_ylabel("compliance")
    axes[3].legend()
    fig.tight_layout()
    fig.savefig("mma_vs_oc.png", dpi=150)
    print("saved mma_vs_oc.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "phase7-mesh-input", "mbb.msh"))