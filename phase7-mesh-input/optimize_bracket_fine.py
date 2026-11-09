"""
The bracket again (pin pulling up, bolt supports, keep rings, void columns),
now with the two scalable pieces so it can run on FINE meshes:

    FastSolver   (fast_solve.py)  AMG-preconditioned CG, warm started
    PDEFilter    (pde_filter.py)  Helmholtz filter, factorized once

Usage:
    python mesh_bracket3d.py 0.06           # ~260k tets
    python optimize_bracket_fine.py bracket.msh

Keep rmin = 0.25 when you refine: rmin sets the smallest feature, so a finer
mesh gives the SAME design with smoother surfaces (mesh independence).
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import meshio

from mesh_io import read_mesh_3d, read_element_sets
from fem3d_solve import precompute_3d
from fem_mesh import compliance_and_sensitivity
from optimize_mesh import oc_update, move_schedule
from optimize_bracket import pin_load, YP
from fast_solve import FastSolver
from pde_filter import PDEFilter


def main(fname, volfrac=0.25, penal=3.0, rmin=0.25, nit=200):
    t0 = time.time()
    nodes, tets, sets, faces = read_mesh_3d(fname)
    esets = read_element_sets(fname)
    keep, void = esets["keep"], esets["void"]
    KE, V, edof, cent = precompute_3d(nodes, tets)
    ndof = 3*len(nodes)
    print(f"mesh: {len(nodes)} nodes, {len(tets)} tets, {ndof} DOFs  "
          f"(element matrices {time.time()-t0:.0f}s)")

    # your BCs and load from the first bracket
    b = sets["bolt"]
    fixed = np.concatenate([3*b, 3*b + 1, 3*b + 2])
    F = pin_load(nodes, faces["pin"], ndof, (0, 1, 0), 1.0, np.array([2.0, YP, 0.0]))

    t1 = time.time()
    solver = FastSolver(KE, edof, ndof, fixed, nodes)
    filt = PDEFilter(nodes, tets, V, rmin)
    print(f"solver + filter setup {time.time()-t1:.0f}s")

    rho = volfrac*np.ones(len(tets))
    rho[keep] = 1.0
    rho[void] = 0.001
    c_hist = []
    for it in range(nit):
        u = solver.solve(rho, penal, F)
        c, dc = compliance_and_sensitivity(rho, penal, KE, edof, u)
        dc = filt.apply(rho, dc, V)
        rho_new = oc_update(rho, V, volfrac, dc, move_schedule(it), keep=keep, void=void)
        change = np.abs(rho_new - rho).max()     # biggest density change
        rho = rho_new

        c_hist.append(c)
        if it % 5 == 0:
            print(f"  it {it:3d}  c={c:8.4f}  change={change:.3f}  "
                  f"CG steps={solver.iters[0]:3d}  ({time.time()-t0:.0f}s)")
        # stop when the DESIGN stops moving, not just the compliance
        if it >= 30 and change < 0.01:
            print(f"  converged at it={it} (max density change < 0.01)")
            break
        if it >= 60 and (max(c_hist[-15:]) - min(c_hist[-15:]))/c < 5e-4:
            print(f"  converged at it={it} (compliance flat for 15 iterations)")
            break

    vol = (rho*V).sum()/V.sum()
    grey = np.sum(V*4*rho*(1 - rho))/V.sum()
    print(f"final: c={c:.4f}  vol={vol:.3f}  greyness={grey:.3f}  "
          f"({time.time()-t0:.0f}s total, {(time.time()-t1)/(it+1):.1f}s per iteration)")

    stem = fname.rsplit(".", 1)[0] + "_fine"
    region = np.zeros(len(tets))
    region[keep] = 1.0
    region[void] = -1.0
    meshio.write(f"{stem}_result.vtu",
                 meshio.Mesh(nodes, [("tetra", tets)],
                             cell_data={"rho": [rho], "region": [region]}))

    s = rho > 0.5
    L, Hh, W = nodes.max(axis=0)
    fig = plt.figure(figsize=(14, 6))
    for k, (elev, azim) in enumerate([(20, -60), (20, 30)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(cent[s, 0], cent[s, 2], cent[s, 1], c=cent[s, 1],
                   cmap="viridis", s=0.5, depthshade=True)
        ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y")
        ax.set_box_aspect((L, W, Hh))
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle(f"bracket, {len(tets)} tets, c = {c:.3f}")
    fig.tight_layout()
    fig.savefig(f"{stem}_result.png", dpi=150, bbox_inches="tight")
    print(f"saved {stem}_result.png and {stem}_result.vtu")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bracket.msh")