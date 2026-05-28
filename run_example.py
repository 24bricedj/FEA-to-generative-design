"""
Driver script for the 1D FEA solver.

Recreates the hand-derived problem from Phase 1:
    L = 3 m, EA = 1 N, 3 elements, b0 = 5 N/m, P = 10 N at x = L,
    u(0) = 0 (fixed).

Expected FEA nodal displacements: [0, 22.5, 40, 52.5]
Analytical: u(x) = -2.5*x^2 + 25*x
"""

import numpy as np

from analytical import u_exact
from plot import plot_displacement_comparison
from solver import (
    apply_dirichlet_bc,
    assemble_load,
    assemble_stiffness,
    build_mesh,
    reconstruct_full_solution,
    solve_system,
)


def main():
    # ---- Problem parameters -------------------------------------------------
    L = 3.0           # bar length [m]
    E = 1.0           # Young's modulus [Pa]
    A = 1.0           # cross-section [m^2]   -> EA = 1 N for easy hand-check
    b0 = 5.0          # uniform body load [N/m]
    P = 10.0          # tip point load [N]
    n_elements = 3

    # ---- Mesh ---------------------------------------------------------------
    nodes, connectivity = build_mesh(L, n_elements)
    print(f"Nodes:        {nodes}")
    print(f"Connectivity: {connectivity.tolist()}")

    # ---- Stiffness ----------------------------------------------------------
    K = assemble_stiffness(nodes, connectivity, E, A)
    print(f"\nGlobal stiffness matrix K:\n{K}")

    # ---- Load vector --------------------------------------------------------
    # Point load P is applied at the last node (x = L)
    point_loads = [(len(nodes) - 1, P)]
    F = assemble_load(nodes, connectivity, b0, point_loads)
    print(f"\nLoad vector F: {F}")

    # ---- Apply BC (u(0) = 0) ------------------------------------------------
    fixed_nodes = [0]
    K_red, F_red, free_nodes = apply_dirichlet_bc(K, F, fixed_nodes)
    print(f"\nReduced K:\n{K_red}")
    print(f"Reduced F: {F_red}")

    # ---- Solve --------------------------------------------------------------
    u_red = solve_system(K_red, F_red)
    u_fea = reconstruct_full_solution(u_red, free_nodes, len(nodes))
    print(f"\nFEA displacements: {u_fea}")

    # ---- Validation against analytical solution -----------------------------
    u_an = u_exact(nodes, E, A, L, b0, P)
    print(f"Analytical:        {u_an}")
    print(f"Nodal error:       {np.abs(u_fea - u_an)}")
    print(f"Max nodal error:   {np.max(np.abs(u_fea - u_an)):.2e}")

    # ---- Plot ---------------------------------------------------------------
    fig, ax = plot_displacement_comparison(
        nodes, u_fea,
        lambda x: u_exact(x, E, A, L, b0, P),
        L,
        savepath="displacement.png",
    )
    print("\nPlot saved to displacement.png")


if __name__ == "__main__":
    main()