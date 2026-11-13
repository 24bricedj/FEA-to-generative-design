"""
Topology optimization of the GE jet engine bracket, scored against the
competition winner (SimJEB 566, M Arie Kurniawan).

    python mesh_ge.py 2.5            # -> ge_envelope.msh
    python optimize_ge.py

Frame (Dolfiny's envelope): mm, N, MPa. Pin along x, vertical +z, out = -y.
Everything here is a piece you built:
    rbe3_force, rbe3_moment_z   (validate_566.py)  loads exactly as SimJEB applied them
    normalized multi-load       (Phase 7)          4 cases, equal weights
    keep, FastSolver, PDEFilter (Phase 7)
"""
import os
import sys
import time
import numpy as np
import meshio

from fem_fast import precompute_3d_fast            # also puts phase7 on the path
from mesh_io import read_mesh_3d, read_element_sets
from fast_solve import FastSolver
from pde_filter import PDEFilter
from fem_mesh import compliance_and_sensitivity
from optimize_mesh import oc_update, move_schedule
from fem3d import iso_D_3d
from validate_566 import rbe3_force, rbe3_moment_z, LBF, INCH, E, NU

WINNER_VOLUME = 77102.1          # mm^3, SimJEB 566 (345 g of Ti-6Al-4V)
WINNER = {"ver": 38062.0, "hor": 22041.4, "dia": 18837.6, "tor": 8740.6}  # N*mm, your validation
WINNER_STRESS = 865.4            # MPa, peak von Mises over the 4 cases (SimJEB)


def von_mises(nodes, tets, u, E, nu):
    """Von Mises stress per element from displacement u (plumbing:
    sigma = D B u_e, the same B you built in Phase 7, all elements at once)."""
    C = np.concatenate([np.ones((len(tets), 4, 1)), nodes[tets]], axis=2)
    G = np.linalg.inv(C)[:, 1:, :]
    ue = u.reshape(-1, 3)[tets]                       # (ne, 4, 3)
    grad = np.einsum('eai,eib->eab', G, ue)           # grad[e, a, b] = d u_b / d x_a
    eps = np.stack([grad[:, 0, 0], grad[:, 1, 1], grad[:, 2, 2],
                    grad[:, 0, 1] + grad[:, 1, 0],
                    grad[:, 1, 2] + grad[:, 2, 1],
                    grad[:, 0, 2] + grad[:, 2, 0]], axis=1)
    s = eps @ iso_D_3d(E, nu).T
    return np.sqrt(0.5*((s[:, 0]-s[:, 1])**2 + (s[:, 1]-s[:, 2])**2 + (s[:, 2]-s[:, 0])**2)
                   + 3*(s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2))


def main(fname="ge_envelope.msh", penal=3.0, nit=200):
    t0 = time.time()
    nodes, tets, sets, faces = read_mesh_3d(fname)
    keep = read_element_sets(fname)["keep"]
    KE, V, edof, tets = precompute_3d_fast(nodes, tets, E, NU)
    ndof = 3*len(nodes)
    bolt, pin = sets["bolt"], sets["pin"]
    fixed = np.concatenate([3*bolt, 3*bolt + 1, 3*bolt + 2])
    print(f"mesh: {len(nodes)} nodes, {len(tets)} tets, envelope {V.sum():.0f} mm^3, "
          f"keep {V[keep].sum():.0f} mm^3  ({time.time()-t0:.0f}s)")

    th = np.radians(42)
    cases = {
        "ver": rbe3_force(nodes, pin, ndof, (0, 0, 8000*LBF)),
        "hor": rbe3_force(nodes, pin, ndof, (0, -8500*LBF, 0)),
        "dia": rbe3_force(nodes, pin, ndof, (0, -9500*LBF*np.sin(th), 9500*LBF*np.cos(th))),
        "tor": rbe3_moment_z(nodes, pin, ndof, 5000*LBF*INCH),
    }
    volfrac = WINNER_VOLUME/V.sum()      # ~0.166: equal mass to the winner
    rmin = 7.5                           # mm: 3 elements of h = 2.5

    names = list(cases)
    Fs = np.column_stack([cases[n] for n in names])
    w = np.ones(len(names))
    print(f"volfrac = {volfrac:.4f}  (target volume {volfrac*V.sum():.0f} mm^3), rmin = {rmin} mm")

    solver = FastSolver(KE, edof, ndof, fixed, nodes)
    filt = PDEFilter(nodes, tets, V, rmin)
    print(f"solver + filter setup done ({time.time()-t0:.0f}s)")

    rho = volfrac*np.ones(len(tets))
    rho[keep] = 1.0
    c0 = None
    c_hist = []
    for it in range(nit):
        U = solver.solve(rho, penal, Fs)
        cs, dcs = [], []
        for i in range(len(names)):
            ci, dci = compliance_and_sensitivity(rho, penal, KE, edof, U[:, i])
            cs.append(ci)
            dcs.append(dci)
        cs, dcs = np.array(cs), np.array(dcs)
        if c0 is None:
            c0 = cs.copy()
        c = np.sum(w*cs/c0)                               # your normalized objective
        dc = ((w/c0)[:, None]*dcs).sum(axis=0)
        dc = filt.apply(rho, dc, V)
        rho_new = oc_update(rho, V, volfrac, dc, move_schedule(it), keep=keep)
        change = np.abs(rho_new - rho).max()
        rho = rho_new
        if it % 5 == 0:
            ratios = "  ".join(f"{n}={r:.3f}" for n, r in zip(names, cs/c0))
            print(f"  it {it:3d}  c={c:.4f}  [{ratios}]  change={change:.3f}  "
                  f"CG={solver.iters}  ({time.time()-t0:.0f}s)")
        c_hist.append(c)
        if it >= 30 and change < 0.01:
            print(f"  converged at it={it} (max density change < 0.01)")
            break
        if it >= 40 and (max(c_hist[-10:]) - min(c_hist[-10:]))/c < 1e-3:
            print(f"  converged at it={it} (objective flat for 10 iterations)")
            break

    # ---------------- score it like a real part ----------------
    # 1. threshold to solid/void at EXACTLY the winner's volume (equal mass)
    order = np.argsort(-rho)
    n_solid = np.searchsorted(np.cumsum(V[order]), WINNER_VOLUME) + 1
    solid = np.zeros(len(tets), bool)
    solid[order[:n_solid]] = True
    solid[keep] = True
    x = np.where(solid, 1.0, 1e-6)                        # void ~ empty
    U = solver.solve(x, 1.0, Fs)
    print(f"\nthresholded design: {V[solid].sum():.0f} mm^3 "
          f"({V[solid].sum()*4.47e-3:.0f} g of Ti-6Al-4V)  vs winner {WINNER_VOLUME:.0f} mm^3 (345 g)\n")
    print(f"{'case':5s} {'yours (N mm)':>14s} {'winner (N mm)':>14s} {'yours/winner':>13s}")
    peak = 0.0
    for i, n in enumerate(names):
        ci = Fs[:, i] @ U[:, i]
        vm = von_mises(nodes, tets, U[:, i], E, NU)[solid]
        peak = max(peak, np.percentile(vm, 99.9))
        print(f"{n:5s} {ci:14.1f} {WINNER[n]:14.1f} {ci/WINNER[n]:12.2f}x")
    print(f"\npeak von Mises (99.9th percentile of solid elements): {peak:.0f} MPa"
          f"   winner: {WINNER_STRESS:.0f} MPa   yield: 903 MPa")

    meshio.write("ge_result.vtu", meshio.Mesh(nodes, [("tetra", tets)],
                 cell_data={"rho": [rho], "solid": [solid.astype(float)],
                            "keep": [np.isin(np.arange(len(tets)), keep).astype(float)]}))
    print(f"saved ge_result.vtu  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()