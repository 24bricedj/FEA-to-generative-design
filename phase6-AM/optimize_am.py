"""MBB beam with and without the AM overhang constraint."""
import numpy as np
import matplotlib.pyplot as plt
from topopt_fast import (element_stiffness_Q4, build_edof, build_filter,
                         fea_solve, compliance_and_sensitivity, apply_filter,
                         mbb_boundary_conditions)
from amfilter import overhang_filter, overhang_backward


def oc_update(nelx, nely, rho, volfrac, dc, move):
    """Phase 5 OC update, with the move limit passed in."""
    l1, l2 = 0.0, 1e9
    while (l2 - l1)/(l1 + l2 + 1e-30) > 1e-4:
        lmid = 0.5*(l1 + l2)
        B = np.maximum(-dc/lmid, 1e-12)
        rho_new = np.maximum(0.001,
                     np.maximum(rho - move,
                        np.minimum(1.0,
                           np.minimum(rho + move,
                              rho*np.sqrt(B)))))
        if rho_new.sum() > volfrac*nelx*nely:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new


# --- CHANGE 1: module-level helper, sits next to the other functions ---
def move_schedule(it):
    """Progressively tighten the move limit so the design settles."""
    if it < 60:  return 0.20
    if it < 150: return 0.10
    if it < 280: return 0.05
    if it < 420: return 0.02
    return 0.01


def run(use_am, nelx=120, nely=40, volfrac=0.5, penal=3.0, rmin=3.0,
        q=8.0, nit=700):                      # --- CHANGE 2a: nit 250 -> 700
    KE = element_stiffness_Q4()
    edof = build_edof(nelx, nely)
    F, fixed = mbb_boundary_conditions(nelx, nely)
    H, Hs = build_filter(nelx, nely, rmin)
    rho = volfrac*np.ones((nely, nelx))
    c_hist = []                                # --- CHANGE 3a: track compliance

    for it in range(nit):
        move = move_schedule(it)               # --- CHANGE 2b: use the schedule

        # --- HOOK 1: design density -> printable density ---
        if use_am:
            rho_hat, S, T, R = overhang_filter(rho, q)
        else:
            rho_hat = rho

        # --- physics runs on the PRINTABLE density ---
        u = fea_solve(nelx, nely, rho_hat, penal, KE, edof, fixed, F)
        c, dc_hat = compliance_and_sensitivity(nelx, nely, rho_hat, penal,
                                               KE, edof, u)
        dc_hat = apply_filter(H, Hs, rho_hat, dc_hat)

        # --- HOOK 2: chain the gradient back to the design variables ---
        if use_am:
            dc = overhang_backward(rho, rho_hat, S, T, R, dc_hat, q)
        else:
            dc = dc_hat

        rho_old = rho.copy()
        rho = oc_update(nelx, nely, rho, volfrac, dc, move)
        change = np.abs(rho - rho_old).max()

        if it % 75 == 0:
            print(f"  it {it:3d}  move={move:.2f}  c={c:8.2f}  "
                  f"vol={rho_hat.mean():.3f}  change={change:.4f}")

        # --- CHANGE 3b: converge on compliance, not max element change ---
        c_hist.append(c)
        if it > 450 and len(c_hist) > 20:
            recent = c_hist[-20:]
            if (max(recent) - min(recent))/abs(c) < 1e-3:
                print(f"  converged at it={it}")
                break

    print(f"  done at it={it}, c={c:.2f}")
    return (rho_hat if use_am else rho), c


def count_violations(r, thr=0.5):
    """Solid elements with no solid support in the three cells below."""
    s = r > thr
    v = 0
    for n in range(r.shape[0]-1):
        below = s[n+1, :]
        L = np.concatenate([[False], below[:-1]])
        R = np.concatenate([below[1:], [False]])
        v += np.sum(s[n, :] & ~(L | below | R))
    return int(v), int(np.sum(s))


def greyness(r):
    """0 = perfectly crisp 0/1 design, higher = more intermediate density."""
    return float(np.mean(4*r*(1-r)))


def main():
    print("=== unconstrained ===")
    r0, c0 = run(False)
    print("=== AM overhang-constrained ===")
    r1, c1 = run(True)

    print(f"\ncompliance: {c0:.1f} -> {c1:.1f}  ({100*(c1-c0)/c0:+.1f}%)")
    v0, n0 = count_violations(r0)
    v1, n1 = count_violations(r1)
    print(f"overhang violations: {v0}/{n0} solid  ->  {v1}/{n1} solid")
    print(f"greyness: {greyness(r0):.3f} -> {greyness(r1):.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 5.4))
    for ax, r, t in zip(axes, [r0, r1],
                        [f"Unconstrained SIMP (c = {c0:.1f})",
                         f"AM overhang-constrained, 45° (c = {c1:.1f})"]):
        full = np.concatenate([np.fliplr(r), r], axis=1)     # mirror to full span
        im = ax.imshow(full, cmap="jet", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02, label="density ρ")
    fig.savefig("am_comparison.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()