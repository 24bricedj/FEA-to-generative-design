"""
Driver for the 1D thermal solver. Validates against the analytical profiles.

Case A: no source, ends fixed at T0 and TL -> linear profile.
Case B: uniform source Q0, same end temps     -> quadratic profile.
"""

import numpy as np

from plot import plot_temperature_profile
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
    """Run the full thermal pipeline, return (nodes, T_fea)."""
    nodes, connectivity = build_mesh(L, n_elements)
    n_nodes = len(nodes)

    K = assemble_conductivity(nodes, connectivity, k, A)
    F = assemble_load(nodes, connectivity, Q0)

    # fix both ends: node 0 -> T0, last node -> TL
    fixed = {0: T0, n_nodes - 1: TL}

    K_red, F_red, free_nodes = apply_dirichlet_bc(K, F, fixed)
    T_red = solve_system(K_red, F_red)
    T_fea = reconstruct_full_solution(T_red, free_nodes, n_nodes, fixed)
    return nodes, T_fea


def main():
    # ---- Problem parameters -------------------------------------------------
    L = 1.0
    k = 1.0
    A = 1.0
    T0 = 100.0     # hot end
    TL = 20.0      # cold end
    n_elements = 4

    # ---- Case A: no source --------------------------------------------------
    print("=== Case A: no source (linear profile) ===")
    Q0 = 0.0
    nodes, T_fea = solve_thermal(L, k, A, T0, TL, Q0, n_elements)
    T_an = T_exact(nodes, k, A, L, T0, TL, Q0)
    print(f"Nodes:       {nodes}")
    print(f"FEA:         {T_fea}")
    print(f"Analytical:  {T_an}")
    print(f"Max error:   {np.max(np.abs(T_fea - T_an)):.2e}")

    # ---- Case B: uniform source ---------------------------------------------
    print("\n=== Case B: uniform source (quadratic profile) ===")
    Q0 = 50.0
    nodes, T_fea = solve_thermal(L, k, A, T0, TL, Q0, n_elements)
    T_an = T_exact(nodes, k, A, L, T0, TL, Q0)
    print(f"Nodes:       {nodes}")
    print(f"FEA:         {T_fea}")
    print(f"Analytical:  {T_an}")
    print(f"Max error:   {np.max(np.abs(T_fea - T_an)):.2e}")



    # re-solve both cases for plotting
    nodes, T_fea_A = solve_thermal(L, k, A, T0, TL, 0.0, n_elements)
    plot_temperature_profile(nodes, T_fea_A,
                             lambda x: T_exact(x, k, A, L, T0, TL, 0.0),
                             L, "Case A: no source (linear)",
                             savepath="profile_caseA.png")

    nodes, T_fea_B = solve_thermal(L, k, A, T0, TL, 50.0, n_elements)
    plot_temperature_profile(nodes, T_fea_B,
                             lambda x: T_exact(x, k, A, L, T0, TL, 50.0),
                             L, "Case B: uniform source (quadratic)",
                             savepath="profile_caseB.png")
    print("\nProfile plots saved.")


if __name__ == "__main__":
    main()