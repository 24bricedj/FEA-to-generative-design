"""
Convergence study for the 1D thermal solver (Case B, quadratic profile).

Like Phase 1: the true profile is quadratic, linear elements give O(h^2)
interpolation error between nodes. Measured against the analytical solution
on a fine grid, the log-log slope should be ~2.
"""

import matplotlib.pyplot as plt
import numpy as np

from analytical import T_exact
from solver import (
    apply_dirichlet_bc,
    assemble_conductivity,
    assemble_load,
    build_mesh,
    reconstruct_full_solution,
    solve_system,
)


def solve_thermal(L, k, A, T0, TL, Q0, n_elements):
    nodes, connectivity = build_mesh(L, n_elements)
    n_nodes = len(nodes)
    K = assemble_conductivity(nodes, connectivity, k, A)
    F = assemble_load(nodes, connectivity, Q0)
    fixed = {0: T0, n_nodes - 1: TL}
    K_red, F_red, free_nodes = apply_dirichlet_bc(K, F, fixed)
    T_red = solve_system(K_red, F_red)
    T_fea = reconstruct_full_solution(T_red, free_nodes, n_nodes, fixed)
    return nodes, T_fea


def evaluate_fea_solution(x_query, nodes, T_nodal):
    """Piecewise-linear interpolation of the FEA solution at query points."""
    return np.interp(x_query, nodes, T_nodal)


def main():
    # ---- Problem parameters (Case B) ----------------------------------------
    L = 1.0
    k = 1.0
    A = 1.0
    T0 = 100.0
    TL = 20.0
    Q0 = 50.0     # uniform source -> quadratic profile

    element_counts = [2, 4, 8, 16, 32, 64, 128]

    x_fine = np.linspace(0, L, 2000)
    T_an_fine = T_exact(x_fine, k, A, L, T0, TL, Q0)

    h_values = []
    errors = []

    print(f"{'n_elem':>8} {'h':>12} {'max error':>15}")
    print("-" * 37)
    for n_elements in element_counts:
        nodes, T_fea = solve_thermal(L, k, A, T0, TL, Q0, n_elements)
        T_fea_fine = evaluate_fea_solution(x_fine, nodes, T_fea)
        max_error = np.max(np.abs(T_fea_fine - T_an_fine))
        h = L / n_elements
        h_values.append(h)
        errors.append(max_error)
        print(f"{n_elements:>8} {h:>12.6f} {max_error:>15.6e}")

    h_values = np.array(h_values)
    errors = np.array(errors)

    log_h = np.log(h_values)
    log_e = np.log(errors)
    slope, _ = np.polyfit(log_h, log_e, 1)
    print(f"\nObserved convergence rate (log-log slope): {slope:.3f}")
    print("(Theory predicts 2.0 for linear elements.)")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(h_values, errors, "o-", color="C3", lw=1.5, ms=8,
              label="Measured error")
    ref = errors[0] * (h_values / h_values[0]) ** 2
    ax.loglog(h_values, ref, "k--", lw=1.0, label="Slope 2 (theory)")
    ax.set_xlabel("Element size h")
    ax.set_ylabel("Max interpolation error [K]")
    ax.set_title(f"1D thermal convergence (slope = {slope:.2f})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig("convergence.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to convergence.png")


if __name__ == "__main__":
    main()