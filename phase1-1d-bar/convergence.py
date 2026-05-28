"""
Convergence study for the 1D FEA solver.

Runs the solver at successively finer meshes and measures the interpolation
error (max deviation between the FEA piecewise-linear solution and the smooth
analytical solution, sampled on a fine grid).

For linear elements, theory predicts the error scales as O(h^2), which shows
up as a straight line of slope 2 on a log-log plot of error vs h.
"""

import matplotlib.pyplot as plt
import numpy as np

from analytical import u_exact
from solver import (
    apply_dirichlet_bc,
    assemble_load,
    assemble_stiffness,
    build_mesh,
    evaluate_fea_solution,
    reconstruct_full_solution,
    solve_system,
)


def solve_fea(L, E, A, b0, P, n_elements):
    """Run the full FEA pipeline and return (nodes, u_fea)."""
    nodes, connectivity = build_mesh(L, n_elements)
    K = assemble_stiffness(nodes, connectivity, E, A)
    point_loads = [(len(nodes) - 1, P)]
    F = assemble_load(nodes, connectivity, b0, point_loads)
    K_red, F_red, free_nodes = apply_dirichlet_bc(K, F, [0])
    u_red = solve_system(K_red, F_red)
    u_fea = reconstruct_full_solution(u_red, free_nodes, len(nodes))
    return nodes, u_fea


def main():
    # ---- Problem parameters (same as run_example.py) ------------------------
    L = 3.0
    E = 1.0
    A = 1.0
    b0 = 5.0
    P = 10.0

    # ---- Mesh refinement levels ---------------------------------------------
    element_counts = [2, 4, 8, 16, 32, 64, 128]

    # Fine grid for measuring interpolation error
    x_fine = np.linspace(0, L, 2000)
    u_an_fine = u_exact(x_fine, E, A, L, b0, P)

    h_values = []
    errors = []

    print(f"{'n_elem':>8} {'h':>12} {'max error':>15}")
    print("-" * 37)
    for n_elements in element_counts:
        nodes, u_fea = solve_fea(L, E, A, b0, P, n_elements)

        # Evaluate FEA solution on the fine grid
        u_fea_fine = evaluate_fea_solution(x_fine, nodes, u_fea)

        # Max interpolation error (L-infinity norm)
        max_error = np.max(np.abs(u_fea_fine - u_an_fine))

        h = L / n_elements
        h_values.append(h)
        errors.append(max_error)

        print(f"{n_elements:>8} {h:>12.6f} {max_error:>15.6e}")

    h_values = np.array(h_values)
    errors = np.array(errors)

    # ---- Measure the observed convergence rate ------------------------------
    # Slope on a log-log plot = convergence order.
    # Fit a line to log(error) vs log(h); the slope is the order.
    log_h = np.log(h_values)
    log_e = np.log(errors)
    slope, intercept = np.polyfit(log_h, log_e, 1)
    print(f"\nObserved convergence rate (log-log slope): {slope:.3f}")
    print("(Theory predicts 2.0 for linear elements.)")

    # ---- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(h_values, errors, "o-", color="C0", lw=1.5, ms=8,
              label="Measured error")

    # Reference slope-2 line for comparison
    ref = errors[0] * (h_values / h_values[0]) ** 2
    ax.loglog(h_values, ref, "k--", lw=1.0, label="Slope 2 (theory)")

    ax.set_xlabel("Element size h [m]")
    ax.set_ylabel("Max interpolation error [m]")
    ax.set_title(f"Mesh convergence (observed slope = {slope:.2f})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.savefig("convergence.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to convergence.png")


if __name__ == "__main__":
    main()