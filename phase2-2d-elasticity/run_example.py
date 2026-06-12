"""
Driver for the 2D cantilever benchmark.

Fully-clamped left edge, distributed downward load on the right edge.
Solves with CST elements and compares tip deflection to Euler-Bernoulli theory.
"""

import numpy as np

from plot import plot_deformed_mesh
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
    """Return the indices of all nodes whose x-coordinate equals x_target."""
    return np.where(np.abs(nodes[:, 0] - x_target) < tol)[0]


def main():
    # ---- Problem parameters -------------------------------------------------
    h = 1.0           # beam depth (in-plane height)
    L = 5.0 * h       # length = 5x depth (moderate aspect ratio)
    t = 1.0           # thickness (out-of-plane)
    E = 1000.0        # Young's modulus
    nu = 0.3          # Poisson's ratio
    P = 1.0           # total tip load (downward)

    nx = 80           # elements along length
    ny = 16           # elements through depth

    # ---- Mesh ---------------------------------------------------------------
    nodes, connectivity = build_mesh(L, h, nx, ny)
    n_dofs = 2 * len(nodes)

    # ---- Assemble stiffness -------------------------------------------------
    K = assemble_stiffness(nodes, connectivity, E, nu, t)

    # ---- Boundary conditions: clamp the left edge (x = 0) -------------------
    left_nodes = nodes_on_vertical_edge(nodes, 0.0)
    fixed_dofs = []
    for n in left_nodes:
        fixed_dofs.append(2 * n)        # u (horizontal)
        fixed_dofs.append(2 * n + 1)    # v (vertical)

    # ---- Load: distribute P downward across the right edge (x = L) ----------
    right_nodes = nodes_on_vertical_edge(nodes, L)
    load_per_node = -P / len(right_nodes)   # negative = downward
    point_loads = [(2 * n + 1, load_per_node) for n in right_nodes]
    F = assemble_load(n_dofs, point_loads)

    # ---- Apply BCs and solve ------------------------------------------------
    K_red, F_red, free_dofs = apply_dirichlet_bc(K, F, fixed_dofs)
    u_red = solve_system(K_red, F_red)
    u_full = reconstruct_full_solution(u_red, free_dofs, n_dofs)

    # ---- Extract tip deflection ---------------------------------------------
    # Average the vertical displacement of the right-edge nodes.
    tip_v = np.mean([u_full[2 * n + 1] for n in right_nodes])
    tip_deflection_fea = abs(tip_v)

    # ---- Compare to beam theory ---------------------------------------------
    delta_beam = cantilever_tip_deflection(P, L, E, h, t)

    print(f"Mesh: {nx} x {ny}  ({len(nodes)} nodes, {len(connectivity)} elements)")
    print(f"FEA tip deflection:        {tip_deflection_fea:.6e}")
    print(f"Euler-Bernoulli deflection: {delta_beam:.6e}")
    print(f"Ratio (FEA / beam):        {tip_deflection_fea / delta_beam:.4f}")

    plot_deformed_mesh(nodes, connectivity, u_full, scale=1.0, savepath="deformed.png")
    print("Deformed mesh saved to deformed.png")

if __name__ == "__main__":
    main()