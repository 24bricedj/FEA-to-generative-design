"""
Convergence study for the 2D conduction solver (sinusoidal-edge benchmark).

The exact solution is curved (sin * sinh), so linear CST elements have real
interpolation error that should converge at O(h^2). Error is measured at the
mesh nodes against the analytical solution; the log-log slope vs. element
size h should be ~2.
"""
import matplotlib.pyplot as plt
import numpy as np

from analytical import T_exact
from solver import (
    apply_dirichlet_bc, assemble_conductivity, build_mesh,
    reconstruct_full_solution, solve_system,
)


def solve_plate(W, H, k, nx, ny):
    nodes, conn = build_mesh(W, H, nx, ny)
    n = len(nodes)
    K = assemble_conductivity(nodes, conn, k)
    F = np.zeros(n)

    # all four edges are Dirichlet; bottom = sin(pi x / W), others = 0
    fixed = {}
    tol = 1e-9
    for idx, (x, y) in enumerate(nodes):
        on_bottom = abs(y) < tol
        on_top = abs(y - H) < tol
        on_left = abs(x) < tol
        on_right = abs(x - W) < tol
        if on_bottom:
            fixed[idx] = np.sin(np.pi * x / W)
        elif on_top or on_left or on_right:
            fixed[idx] = 0.0

    K_red, F_red, free = apply_dirichlet_bc(K, F, fixed)
    T_red = solve_system(K_red, F_red)
    T_fea = reconstruct_full_solution(T_red, free, n, fixed)
    T_an = T_exact(nodes[:, 0], nodes[:, 1], W, H)
    max_err = np.max(np.abs(T_fea - T_an))
    return max_err


def main():
    W, H, k = 1.0, 1.0, 1.0
    mesh_seq = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]

    hs, errs = [], []
    print(f"{'nx x ny':>10} {'h':>10} {'max error':>15}")
    print("-" * 38)
    for nx, ny in mesh_seq:
        err = solve_plate(W, H, k, nx, ny)
        h = W / nx
        hs.append(h); errs.append(err)
        print(f"{f'{nx}x{ny}':>10} {h:>10.5f} {err:>15.6e}")

    hs, errs = np.array(hs), np.array(errs)
    slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)
    print(f"\nObserved convergence rate (log-log slope): {slope:.3f}")
    print("(Theory predicts 2.0 for linear elements.)")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(hs, errs, "o-", color="C3", lw=1.5, ms=8, label="Measured error")
    ref = errs[0] * (hs / hs[0]) ** 2
    ax.loglog(hs, ref, "k--", lw=1.0, label="Slope 2 (theory)")
    ax.set_xlabel("Element size h")
    ax.set_ylabel("Max nodal error [K]")
    ax.set_title(f"2D conduction convergence (slope = {slope:.2f})")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.savefig("convergence.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to convergence.png")


if __name__ == "__main__":
    main()