"""
Prove the stress gradient is right BEFORE optimizing with it.

    python mesh_lbracket.py 4        # coarse mesh, quick
    python check_stress_grad.py lbracket.msh

Two checks, from the inside out:

 1. d sig_PN / du  (your JOBs 1-3) against finite differences in u:
        [sig_PN(u + h e_i) - sig_PN(u - h e_i)] / 2h       for a few DOFs i
 2. the full chain  rho -> filter -> K u = F -> sig_PN  (adds JOB 4, the adjoint
    solve and the filter transpose) against finite differences in rho:
        [sig_PN(rho + h e_j) - sig_PN(rho - h e_j)] / 2h   for a few elements j
    Each of those costs two full FEA solves -- the adjoint gets ALL of them
    from one extra solve. That's the whole point of the adjoint method.

Central differences are accurate to ~h^2, so expect agreement to 1e-5 or better.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "phase7-mesh-input"))
from mesh_io import read_mesh                                     # noqa: E402
from fem_mesh import precompute                                   # noqa: E402
from density_filter import DensityFilter                          # noqa: E402
from fem2d_solve import Solver2D                                  # noqa: E402
from stress2d import StressMeasure, adjoint_term, cst_B, plane_stress_D   # noqa: E402

E, NU, PENAL = 70e3, 0.3, 3.0


def setup(fname, rmin):
    nodes, tris, sets = read_mesh(fname)
    KE, V, edof, cent = precompute(nodes, tris, E, NU)
    ndof = 2*len(nodes)
    fixed = np.concatenate([2*sets["fixed"], 2*sets["fixed"] + 1])
    F = np.zeros(ndof)
    F[2*sets["load"] + 1] = -100.0/len(sets["load"])
    sm = StressMeasure(cst_B(nodes, tris), plane_stress_D(E, NU), edof, ndof)
    return nodes, tris, KE, V, edof, ndof, fixed, F, DensityFilter(cent, V, rmin), sm


def main(fname="lbracket.msh", rmin=8.0):
    nodes, tris, KE, V, edof, ndof, fixed, F, df, sm = setup(fname, rmin)
    solver = Solver2D(KE, edof, ndof, fixed)
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.3, 1.0, len(tris))
    print(f"{len(tris)} elements, {ndof} DOFs\n")

    def pn_of_rho(r):
        rt = df.forward(r)
        solver.factor(rt, PENAL)
        return sm.evaluate(rt, solver.solve(F))[0]

    # ---- forward + adjoint, once
    rho_t = df.forward(rho)
    solver.factor(rho_t, PENAL)
    u = solver.solve(F)
    pn, sig, vm, dpn_du, direct = sm.evaluate(rho_t, u)
    lam = solver.solve(dpn_du)                                   # the adjoint solve
    dpn_t = direct + adjoint_term(rho_t, PENAL, KE, edof, u, lam)
    dpn = df.backward(dpn_t)
    print(f"sig_PN = {pn:.4f} MPa,  max element sig = {sig.max():.4f} MPa  "
          f"(max/PN = {sig.max()/pn:.3f}: the p-norm overestimates the max, as derived)\n")

    # ---- check 1: d sig_PN / du
    print("check 1: d sig_PN/du   (JOBs 1-3)")
    free = np.setdiff1d(np.arange(ndof), fixed)
    dofs = free[np.argsort(-np.abs(dpn_du[free]))[:3]]           # the 3 most sensitive
    dofs = np.concatenate([dofs, rng.choice(free, 3, replace=False)])
    h = 1e-6*np.abs(u).max()
    worst1 = 0.0
    for i in dofs:
        up, um = u.copy(), u.copy()
        up[i] += h
        um[i] -= h
        fd = (sm.evaluate(rho_t, up)[0] - sm.evaluate(rho_t, um)[0])/(2*h)
        err = abs(fd - dpn_du[i])/max(abs(fd), 1e-12*abs(dpn_du).max())
        worst1 = max(worst1, err if abs(fd) > 1e-8*np.abs(dpn_du).max() else 0.0)
        print(f"  dof {i:5d}: analytic {dpn_du[i]: .6e}   finite diff {fd: .6e}   rel err {err:.1e}")

    # ---- check 2: the full chain
    print("\ncheck 2: d sig_PN/d rho through filter + FEA   (adds JOB 4)")
    els = np.concatenate([np.argsort(-np.abs(dpn))[:3], rng.choice(len(tris), 3, replace=False)])
    h = 1e-5
    worst2 = 0.0
    for j in els:
        rp, rm = rho.copy(), rho.copy()
        rp[j] += h
        rm[j] -= h
        fd = (pn_of_rho(rp) - pn_of_rho(rm))/(2*h)
        err = abs(fd - dpn[j])/max(abs(fd), 1e-12)
        if abs(fd) > 1e-6*np.abs(dpn).max():
            worst2 = max(worst2, err)
        print(f"  elem {j:5d}: adjoint {dpn[j]: .6e}   finite diff {fd: .6e}   rel err {err:.1e}")

    ok = worst1 < 1e-4 and worst2 < 1e-4
    print(f"\nworst relative error: {worst1:.1e} (du), {worst2:.1e} (drho)  ->  "
          f"{'PASS' if ok else 'FAIL -- check the JOB lines'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lbracket.msh")