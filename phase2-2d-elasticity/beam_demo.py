"""
Application demo (not a validation): a 2D beam fixed at BOTH ends with a
downward point load at the center. Aluminum material properties.

Uses the Phase 2 elasticity solver completely unchanged -- this is just a
different geometry, material, load, and set of boundary conditions. The point
is to show the solver is a general 2D elasticity tool, not a one-off cantilever
demo: feed it a new problem and it computes the real deformation.
"""

import numpy as np

from plot import plot_deformed_mesh
from solver import (
    apply_dirichlet_bc,
    assemble_load,
    assemble_stiffness,
    build_mesh,
    reconstruct_full_solution,
    solve_system,
)


def nodes_on_vertical_edge(nodes, x_target, tol=1e-9):
    """Indices of nodes whose x-coordinate equals x_target."""
    return np.where(np.abs(nodes[:, 0] - x_target) < tol)[0]


def node_nearest(nodes, x, y):
    """Index of the node closest to (x, y)."""
    return np.argmin((nodes[:, 0] - x) ** 2 + (nodes[:, 1] - y) ** 2)


def main():
    # ---- Aluminum, real-ish units (MPa, mm, N) ------------------------------
    E = 69000.0    # MPa  (69 GPa, aluminum)
    nu = 0.33      # aluminum Poisson's ratio
    t = 10.0       # mm thickness (out-of-plane)

    L = 200.0      # mm span
    H = 20.0       # mm depth
    nx, ny = 80, 8
    P = 10000.0     # N total downward load at center

    # ---- Mesh ---------------------------------------------------------------
    nodes, conn = build_mesh(L, H, nx, ny)
    ndof = 2 * len(nodes)

    # ---- Stiffness ----------------------------------------------------------
    K = assemble_stiffness(nodes, conn, E, nu, t)

    # ---- BCs: fix BOTH ends (x=0 and x=L), both DOFs ------------------------
    left = nodes_on_vertical_edge(nodes, 0.0)
    right = nodes_on_vertical_edge(nodes, L)
    fixed = []
    for n in np.concatenate([left, right]):
        fixed.append(2 * n)        # u (horizontal)
        fixed.append(2 * n + 1)    # v (vertical)

    # ---- Load: downward point load at the top-center node -------------------
    cnode = node_nearest(nodes, L / 2.0, H)
    F = assemble_load(ndof, [(2 * cnode + 1, -P)])

    # ---- Solve --------------------------------------------------------------
    Kr, Fr, free = apply_dirichlet_bc(K, F, fixed)
    ur = solve_system(Kr, Fr)
    u = reconstruct_full_solution(ur, free, ndof)

    # ---- Report deflection --------------------------------------------------
    v = u.reshape(-1, 2)[:, 1]
    print(f"Aluminum fixed-fixed beam, center load {P} N")
    print(f"Span {L} mm, depth {H} mm, E = {E} MPa")
    print(f"Max vertical deflection (FEA): {np.min(v):.4f} mm")

    # beam-theory reference for a fixed-fixed beam with a center point load:
    #     delta = P L^3 / (192 E I)
    I = t * H**3 / 12.0
    d_beam = P * L**3 / (192.0 * E * I)
    print(f"Beam-theory (fixed-fixed) deflection: {d_beam:.4f} mm")

    # ---- Plot the deformed shape (exaggerated) ------------------------------
    plot_deformed_mesh(nodes, conn, u, scale=5.0,
                       savepath="beam_fixedfixed.png")
    print("Saved beam_fixedfixed.png")


if __name__ == "__main__":
    main()