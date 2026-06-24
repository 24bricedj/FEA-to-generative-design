"""
Driver for the 2D sinusoidal-edge conduction benchmark.

Plate [0,W]x[0,H], Laplace (no source):
    bottom edge: T = sin(pi x / W)
    top/left/right edges: T = 0
Solves with CST elements, validates against the exact sin*sinh solution,
and plots the temperature field.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from analytical import T_exact
from solver import (
    apply_dirichlet_bc, assemble_conductivity, build_mesh,
    reconstruct_full_solution, solve_system,
)


def main():
    W, H, k = 1.0, 1.0, 1.0
    nx, ny = 32, 32

    nodes, conn = build_mesh(W, H, nx, ny)
    n = len(nodes)
    K = assemble_conductivity(nodes, conn, k)
    F = np.zeros(n)

    fixed = {}
    tol = 1e-9
    for idx, (x, y) in enumerate(nodes):
        if abs(y) < tol:
            fixed[idx] = np.sin(np.pi * x / W)
        elif abs(y - H) < tol or abs(x) < tol or abs(x - W) < tol:
            fixed[idx] = 0.0

    K_red, F_red, free = apply_dirichlet_bc(K, F, fixed)
    T_red = solve_system(K_red, F_red)
    T_fea = reconstruct_full_solution(T_red, free, n, fixed)

    T_an = T_exact(nodes[:, 0], nodes[:, 1], W, H)
    max_err = np.max(np.abs(T_fea - T_an))
    print(f"Mesh: {nx} x {ny}  ({n} nodes, {len(conn)} elements)")
    print(f"Max nodal error vs. analytical: {max_err:.2e}")

    # temperature field plot
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], conn)
    fig, ax = plt.subplots(figsize=(7, 6))
    tpc = ax.tripcolor(triang, T_fea, shading="gouraud", cmap="inferno")
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("2D temperature field (sin edge, sinh decay)")
    fig.colorbar(tpc, ax=ax, label="Temperature")
    fig.savefig("temperature_field.png", dpi=150, bbox_inches="tight")
    print("Temperature field saved to temperature_field.png")


if __name__ == "__main__":
    main()