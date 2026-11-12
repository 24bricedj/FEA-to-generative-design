"""
Multi-load 3D SIMP on the bracket: one design that has to handle several
pin loads (GE-bracket style), with normalized weights.

Usage:
    python mesh_bracket3d.py 0.06            # fine mesh, ~260k tets
    python optimize_bracket_multi.py bracket.msh

Pieces:
    pin_load            (yours, optimize_bracket.py)  any in-plane pin direction
    normalized weighted compliance  (yours, below)
    FastSolver          (fast_solve.py)  ONE multigrid setup per iteration,
                        reused for every load case (like factorizing once)
    PDEFilter           (pde_filter.py)  factorized once for the whole run
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import meshio

from mesh_io import read_mesh_3d, read_element_sets
from fem3d_solve import precompute_3d
from fem_mesh import compliance_and_sensitivity
from fast_solve import FastSolver
from pde_filter import PDEFilter
from optimize_mesh import oc_update, move_schedule
from optimize_bracket import pin_load, torsion_load, YP

AXIS_PT = np.array([2.0, YP, 0.0])        # a point on the pin axis

# load cases: (name, unit direction in the x-y plane, weight)
th = np.radians(42)
CASES = [
    ("up",         (0.0, 1.0, 0.0),               1.0),
    ("horizontal", (1.0, 0.0, 0.0),               1.0),
    ("42 deg",     (np.sin(th), np.cos(th), 0.0), 1.0),
]


def main(fname, volfrac=0.25, penal=3.0, rmin=0.25, nit=200):
    t0 = time.time()
    nodes, tets, sets, faces = read_mesh_3d(fname)
    esets = read_element_sets(fname)
    KE, V, edof, cent = precompute_3d(nodes, tets)
    ndof = 3*len(nodes)
    keep, void = esets["keep"], esets["void"]
    print(f"mesh: {len(nodes)} nodes, {len(tets)} tets, {ndof} DOFs  "
          f"(setup {time.time()-t0:.1f}s)")

    b = sets["bolt"]
    fixed = np.concatenate([3*b, 3*b + 1, 3*b + 2])
    names = [n for n, d, w in CASES]
    w = np.array([w for n, d, w in CASES])

    # ------------------------------------------------------------------
    # YOUR JOB, piece 1: the load matrix Fs, shape (ndof, n_cases).
    # Column i is pin_load(...) for case i's direction, with T = 1.0 and
    # axis_pt = AXIS_PT.  Build a Python list of the columns, then stack them:
    #     Fs = np.column_stack(list_of_F_vectors)
    # (a for-loop over CASES, appending to a list, is the simplest way;
    #  each entry of CASES unpacks as  name, d, wi)
    # ------------------------------------------------------------------
    cols = []
    for name, d, wi in CASES:
        cols.append(pin_load(nodes, faces["pin"], ndof, d, 1.0, AXIS_PT))
    cols.append(torsion_load(nodes, faces["pin"], ndof, 1.0, AXIS_PT))   # case 4
    names.append("torsion")
    w = np.append(w, 1.0)                                                # its weight
    Fs = np.column_stack(cols)

    for i, n in enumerate(names):
        print(f"  case {n:11s}: |F| = {np.linalg.norm([Fs[0::3, i].sum(), Fs[1::3, i].sum()]):.3f}")

    rho = volfrac*np.ones(len(tets))
    rho[keep] = 1.0
    rho[void] = 0.001
    t1 = time.time()
    solver = FastSolver(KE, edof, ndof, fixed, nodes)
    filt = PDEFilter(nodes, tets, V, rmin)
    print(f"solver + filter setup {time.time()-t1:.0f}s")
    c_hist = []
    c0 = None

    for it in range(nit):
        U = solver.solve(rho, penal, Fs)
        # per-case compliance and sensitivity (your Phase 5 function, once per column)
        cs, dcs = [], []
        for i in range(Fs.shape[1]):
            ci, dci = compliance_and_sensitivity(rho, penal, KE, edof, U[:, i])
            cs.append(ci)
            dcs.append(dci)
        cs = np.array(cs)                 # (n_cases,)
        dcs = np.array(dcs)               # (n_cases, n_elements)
        if c0 is None:
            c0 = cs.copy()                # reference: each case's iteration-0 value

        # --------------------------------------------------------------
        # YOUR JOB, piece 2: the NORMALIZED weighted objective.
        #     c  = sum_i  w_i * cs_i / c0_i                 (a single number)
        #     dc = sum_i  w_i * dcs_i / c0_i                (one per element)
        # cs, c0, w are (n_cases,) arrays; dcs is (n_cases, n_elements).
        # --------------------------------------------------------------
        c = np.sum(w*cs/c0)
        dc = ((w/c0)[:, None]*dcs).sum(axis=0)

        dc = filt.apply(rho, dc, V)
        rho_new = oc_update(rho, V, volfrac, dc, move_schedule(it),
                            keep=keep, void=void)
        change = np.abs(rho_new - rho).max()
        rho = rho_new

        c_hist.append(c)
        if it % 5 == 0:
            ratios = "  ".join(f"{n}={r:.3f}" for n, r in zip(names, cs/c0))
            print(f"  it {it:3d}  c={c:.4f}  [c_i/c0_i: {ratios}]  "
                  f"change={change:.3f}  CG={solver.iters}  ({time.time()-t0:.0f}s)")
        if it >= 30 and change < 0.01:
            print(f"  converged at it={it} (max density change < 0.01)")
            break
        if it >= 60 and (max(c_hist[-15:]) - min(c_hist[-15:]))/c < 5e-4:
            print(f"  converged at it={it} (objective flat for 15 iterations)")
            break

    vol = (rho*V).sum()/V.sum()
    grey = np.sum(V*4*rho*(1 - rho))/V.sum()
    print(f"final: c={c:.4f}  vol={vol:.3f}  greyness={grey:.3f}  "
          f"({time.time()-t0:.0f}s total)")
    for n, ci in zip(names, cs):
        print(f"  {n:11s}: c = {ci:.3f}")

    stem = fname.rsplit(".", 1)[0] + "_multi"
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
                   cmap="viridis", s=2, depthshade=True)
        ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y")
        ax.set_box_aspect((L, W, Hh))
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle(f"bracket, {len(names)} load cases ({', '.join(names)}), "
                 f"volfrac {volfrac}")
    fig.tight_layout()
    fig.savefig(f"{stem}_result.png", dpi=150, bbox_inches="tight")
    print(f"saved {stem}_result.png and {stem}_result.vtu")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bracket.msh")