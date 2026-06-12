"""
Convergence study for the 2D cantilever benchmark.

Two complementary views, since beam theory is NOT the exact 2D answer:

  Plot A (plateau):   tip deflection vs. mesh refinement, converging to the
                      true 2D value, with the Euler-Bernoulli value drawn as
                      a reference line we approach and slightly pass.

  Plot B (self-conv): |delta(h) - delta_finest| vs. element size h on log-log.
                      Using the finest mesh as a stand-in for the (unknown)
                      exact answer recovers a clean convergence rate, the same
                      way Phase 1 measured error against the analytical solution.
"""

import matplotlib.pyplot as plt
import numpy as np

from analytical import cantilever_tip_deflection
from solver import (
    apply_dirichlet_bc,
    assemble_load,
    assemble_stiffness,
    build_mesh,
    reconstruct_full_solution,
    solve_system,
)


def nodes_on_vertical_edge(nodes, x_target, tol=1e-9):
    """Return indices of all nodes whose x-coordinate equals x_target."""
    return np.where(np.abs(nodes[:, 0] - x_target) < tol)[0]


def solve_cantilever(L, h, t, E, nu, P, nx, ny):
    """Run the full pipeline for one mesh, return the tip deflection magnitude."""
    nodes, connectivity = build_mesh(L, h, nx, ny)
    n_dofs = 2 * len(nodes)

    K = assemble_stiffness(nodes, connectivity, E, nu, t)

    # clamp left edge (x = 0): fix both DOFs
    left_nodes = nodes_on_vertical_edge(nodes, 0.0)
    fixed_dofs = []
    for n in left_nodes:
        fixed_dofs.append(2 * n)
        fixed_dofs.append(2 * n + 1)

    # distribute downward load across right edge (x = L)
    right_nodes = nodes_on_vertical_edge(nodes, L)
    load_per_node = -P / len(right_nodes)
    point_loads = [(2 * n + 1, load_per_node) for n in right_nodes]
    F = assemble_load(n_dofs, point_loads)

    K_red, F_red, free_dofs = apply_dirichlet_bc(K, F, fixed_dofs)
    u_red = solve_system(K_red, F_red)
    u_full = reconstruct_full_solution(u_red, free_dofs, n_dofs)

    tip_v = np.mean([u_full[2 * n + 1] for n in right_nodes])
    return abs(tip_v)


def main():
    # ---- Problem parameters (same as run_example.py) ------------------------
    h = 1.0
    L = 5.0 * h
    t = 1.0
    E = 1000.0
    nu = 0.3
    P = 1.0

    # ---- Mesh refinement sequence (h halves each step) ----------------------
    mesh_sequence = [(10, 2), (20, 4), (40, 8), (80, 16), (160, 32)]

    element_sizes = []   # h = L / nx for each mesh
    deflections = []
    element_counts = []

    print(f"{'nx x ny':>10} {'n_elem':>8} {'h':>10} {'tip defl':>14}")
    print("-" * 46)
    for nx, ny in mesh_sequence:
        delta = solve_cantilever(L, h, t, E, nu, P, nx, ny)
        h_elem = L / nx
        element_sizes.append(h_elem)
        deflections.append(delta)
        element_counts.append(2 * nx * ny)
        print(f"{f'{nx}x{ny}':>10} {2*nx*ny:>8} {h_elem:>10.5f} {delta:>14.6e}")

    element_sizes = np.array(element_sizes)
    deflections = np.array(deflections)
    element_counts = np.array(element_counts)

    delta_beam = cantilever_tip_deflection(P, L, E, h, t)

    # ---- Self-convergence: error vs. finest mesh ----------------------------
    delta_finest = deflections[-1]
    # exclude the finest point itself (its error against itself is 0)
    errors = np.abs(deflections[:-1] - delta_finest)
    h_for_fit = element_sizes[:-1]

    # fit slope on log-log (convergence order)
    log_h = np.log(h_for_fit)
    log_e = np.log(errors)
    slope, _ = np.polyfit(log_h, log_e, 1)
    print(f"\nEuler-Bernoulli reference: {delta_beam:.6e}")
    print(f"Finest-mesh deflection:    {delta_finest:.6e}")
    print(f"Self-convergence rate (log-log slope): {slope:.3f}")

    # ---- Plot A: plateau ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(element_counts, deflections, "o-", color="C0", lw=1.5, ms=8,
            label="FEA tip deflection")
    ax.axhline(delta_beam, color="k", ls="--", lw=1.0,
               label="Euler-Bernoulli")
    ax.set_xscale("log")
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Tip deflection")
    ax.set_title("Cantilever convergence (plateau)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig("convergence_plateau.png", dpi=150, bbox_inches="tight")

    # ---- Plot B: self-convergence rate --------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.loglog(h_for_fit, errors, "o-", color="C1", lw=1.5, ms=8,
               label="|delta(h) - delta_finest|")
    ref = errors[0] * (h_for_fit / h_for_fit[0]) ** 2
    ax2.loglog(h_for_fit, ref, "k--", lw=1.0, label="Slope 2 (reference)")
    ax2.set_xlabel("Element size h")
    ax2.set_ylabel("Deflection error vs. finest mesh")
    ax2.set_title(f"Self-convergence (slope = {slope:.2f})")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    fig2.savefig("convergence_selfconv.png", dpi=150, bbox_inches="tight")

    print("\nPlots saved: convergence_plateau.png, convergence_selfconv.png")


if __name__ == "__main__":
    main()