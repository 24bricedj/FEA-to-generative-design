"""
Stress-constrained topology optimization of the L-bracket, against the
stiffness-optimized design at the SAME volume.

    python mesh_lbracket.py 1.0
    python optimize_stress.py lbracket.msh

  design A:  minimize volume   s.t.  peak von Mises <= SIG_LIM   (MMAConstrained)
  design B:  minimize compliance at design A's volume            (your Phase 9 MMA)

Prediction to write down BEFORE you run it: where does design B's peak stress
sit, and what does design A do differently there?

Units: mm, N, MPa. Aluminum (E = 70 GPa), 1 mm thick plate, 200 N down on
the right edge.
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
from fem_mesh import precompute                                   # noqa: E402
from density_filter import DensityFilter                          # noqa: E402
from fem2d_solve import Solver2D                                  # noqa: E402
from stress2d import StressMeasure, adjoint_term, cst_B, plane_stress_D   # noqa: E402
from mma import MMA                                               # noqa: E402
from mma_constrained import MMAConstrained                        # noqa: E402

E, NU, PENAL = 70e3, 0.3, 3.0
LOAD = 200.0          # N
SIG_LIM = 200.0       # MPa: 6061-T6 yields at ~275, so a 1.4 factor of safety
RMIN = 3.0            # mm


def setup(fname):
    nodes, tris, sets = read_mesh(fname)
    KE, V, edof, cent = precompute(nodes, tris, E, NU)
    ndof = 2*len(nodes)
    fixed = np.concatenate([2*sets["fixed"], 2*sets["fixed"] + 1])
    F = np.zeros(ndof)
    F[2*sets["load"] + 1] = -LOAD/len(sets["load"])
    # keep a small solid pad under the load so the load has something to act on
    keep = np.where((cent[:, 0] > 95) & (cent[:, 1] > 30))[0]
    return dict(nodes=nodes, tris=tris, KE=KE, V=V, edof=edof, cent=cent, ndof=ndof,
                fixed=fixed, F=F, keep=keep,
                df=DensityFilter(cent, V, RMIN),
                solver=Solver2D(KE, edof, ndof, fixed),
                sm=StressMeasure(cst_B(nodes, tris), plane_stress_D(E, NU), edof, ndof))


def analyze(m, rho_t):
    """FEA + stress on the filtered field: compliance, element stresses."""
    m["solver"].factor(rho_t, PENAL)
    u = m["solver"].solve(m["F"])
    pn, sig, vm, dpn_du, direct = m["sm"].evaluate(rho_t, u)
    return u, m["F"] @ u, pn, sig, vm, dpn_du, direct


def stress_design(m, nit=400):
    df, V, keep = m["df"], m["V"], m["keep"]
    n = len(V)
    rho = np.ones(n)                                  # start solid, carve away
    opt = MMAConstrained(n, move=0.05)              # stress is far more nonlinear than compliance: smaller steps
    dv = df.backward(V)/V.sum()                       # gradient of volume fraction
    ck = 1.0
    t0 = time.time()
    hist = []
    for it in range(nit):
        rho_t = df.forward(rho)
        u, c, pn, sig, vm, dpn_du, direct = analyze(m, rho_t)
        lam = m["solver"].solve(dpn_du)               # adjoint: same K, new RHS
        dpn = df.backward(direct + adjoint_term(rho_t, PENAL, m["KE"], m["edof"], u, lam))

        # adaptive correction (your derivation): scale the p-norm so it tracks
        # the real max. Damped -- a jumpy ck makes MMA chase a moving target.
        ck_new = sig.max()/pn
        ck = ck_new if it == 0 else 0.5*ck + 0.5*ck_new

        f1 = ck * pn / SIG_LIM -1
        df1 = ck * dpn / SIG_LIM 


        f0 = (rho_t*V).sum()/V.sum()
        rho_new = opt.update_c(rho, dv, f1, df1, keep=keep)
        change = np.abs(df.forward(rho_new) - rho_t).max()
        rho = rho_new
        hist.append((f0, sig.max()))
        if it % 20 == 0:
            print(f"  stress it {it:3d}  vol={f0:.3f}  max sig={sig.max():6.1f} MPa  "
                  f"PN*ck={ck*pn:6.1f}  change={change:.3f}  ({time.time()-t0:.0f}s)")
        if it >= 50 and change < 0.005 and sig.max() < 1.02*SIG_LIM:
            break
    rho_t = df.forward(rho)
    print(f"stress design: vol = {(rho_t*V).sum()/V.sum():.3f} after {it+1} iterations\n")
    return rho_t, hist


def compliance_design(m, volfrac, nit=300):
    df, V, keep = m["df"], m["V"], m["keep"]
    n = len(V)
    rho = volfrac*np.ones(n)
    rho[keep] = 1.0
    opt = MMA(n)
    dv = df.backward(V)
    t0 = time.time()
    for it in range(nit):
        rho_t = df.forward(rho)
        u, c, *_ = analyze(m, rho_t)
        ue = u[m["edof"]]
        dc_t = -PENAL*rho_t**(PENAL - 1)*np.einsum('ei,eij,ej->e', ue, m["KE"], ue)
        rho_new = opt.update(rho, df.backward(dc_t), dv, volfrac*V.sum(), keep=keep)
        change = np.abs(df.forward(rho_new) - rho_t).max()
        rho = rho_new
        if it % 20 == 0:
            print(f"  compliance it {it:3d}  c={c:9.2f}  change={change:.3f}  ({time.time()-t0:.0f}s)")
        if it >= 30 and change < 0.01:
            break
    print(f"compliance design: c = {c:.2f} after {it+1} iterations\n")
    return df.forward(rho)


def main(fname):
    m = setup(fname)
    print(f"{len(m['tris'])} elements; limit {SIG_LIM} MPa, load {LOAD} N\n")
    rho_A, hist = stress_design(m)
    volA = (rho_A*m["V"]).sum()/m["V"].sum()
    rho_B = compliance_design(m, volA)

    tri = mtri.Triangulation(m["nodes"][:, 0], m["nodes"][:, 1], m["tris"])
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    rows = []
    for k, (name, r) in enumerate([("A: min volume s.t. stress", rho_A),
                                   ("B: min compliance, same volume", rho_B)]):
        u, c, pn, sig, vm, *_ = analyze(m, r)
        solid = r > 0.5
        rows.append((name, (r*m["V"]).sum()/m["V"].sum(), c, sig.max()))
        ax[0, k].tripcolor(tri, facecolors=r, cmap="gray_r", vmin=0, vmax=1)
        ax[0, k].set_title(f"{name}\nvol {rows[-1][1]:.3f},  c = {c:.1f} N mm")
        h = ax[1, k].tripcolor(tri, facecolors=np.where(solid, sig, np.nan), cmap="jet",
                               vmin=0, vmax=max(SIG_LIM*1.5, 1))
        ax[1, k].set_title(f"relaxed von Mises, peak {sig.max():.0f} MPa (limit {SIG_LIM:.0f})")
        fig.colorbar(h, ax=ax[1, k], shrink=0.8, label="MPa")
        for a in ax[:, k]:
            a.set_aspect("equal")
            a.axis("off")
    fig.tight_layout()
    fig.savefig("stress_vs_compliance.png", dpi=150)

    print(f"{'design':34s} {'volume':>7s} {'compliance':>11s} {'peak sigma':>11s}")
    for name, v, c, s in rows:
        print(f"{name:34s} {v:7.3f} {c:11.1f} {s:9.0f} MPa")
    print("saved stress_vs_compliance.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lbracket.msh")