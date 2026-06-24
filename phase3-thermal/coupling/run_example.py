"""
Coupled thermal-structural benchmark: fully-constrained uniform heating.

A block heated uniformly (dT above reference), fully clamped on all four edges
so it cannot expand. The thermal load drives the elasticity solve; the
recovered stress should match the closed-form fully-constrained thermal stress.
"""
import numpy as np

from thermal_load import assemble_thermal_load
from analytical import constrained_thermal_stress
from solver import (
    build_mesh, assemble_stiffness, apply_dirichlet_bc,
    solve_system, reconstruct_full_solution, recover_element_stress,
)


def main():
    E = 200000.0     # MPa  (steel-ish)
    nu = 0.3
    alpha = 12e-6    # /K
    t = 1.0          # mm

    W = H = 1.0
    nx = ny = 10

    T_ref = 0.0
    dT_uniform = 100.0

    nodes, conn = build_mesh(W, H, nx, ny)
    n = len(nodes); ndof = 2*n

    T = np.full(n, dT_uniform)   # uniform temperature field

    K = assemble_stiffness(nodes, conn, E, nu, t)
    F_th = assemble_thermal_load(nodes, conn, T, T_ref, E, nu, alpha, t)

    # fully constrain ALL boundary nodes
    tol = 1e-9
    fixed = []
    for idx, (x, y) in enumerate(nodes):
        if (abs(x) < tol or abs(x-W) < tol or abs(y) < tol or abs(y-H) < tol):
            fixed.append(2*idx); fixed.append(2*idx+1)

    Kr, Fr, free = apply_dirichlet_bc(K, F_th, fixed)
    ur = solve_system(Kr, Fr)
    u = reconstruct_full_solution(ur, free, ndof)

    # recover stress in the element nearest the center
    cx, cy = W/2, H/2
    best, bestd = None, 1e9
    for ei, el in enumerate(conn):
        c = nodes[el].mean(axis=0)
        d = (c[0]-cx)**2 + (c[1]-cy)**2
        if d < bestd: bestd, best = d, ei
    el = conn[best]
    coords = nodes[el]
    u_elem = np.array([u[2*el[0]], u[2*el[0]+1],
                       u[2*el[1]], u[2*el[1]+1],
                       u[2*el[2]], u[2*el[2]+1]])
    dT_elem = T[el].mean() - T_ref
    sigma = recover_element_stress(coords, u_elem, E, nu, alpha, dT_elem)
    sig_exact = constrained_thermal_stress(E, nu, alpha, dT_uniform)

    print(f"Coupled fully-constrained heating, dT = {dT_uniform} K")
    print(f"  FEA stress:   sigma_x={sigma[0]:.4f}, sigma_y={sigma[1]:.4f}, tau_xy={sigma[2]:.4e} MPa")
    print(f"  Analytical:   sigma_x=sigma_y={sig_exact:.4f}, tau_xy=0 MPa")
    print(f"  Error:        {abs(sigma[0]-sig_exact):.4e} MPa")


if __name__ == "__main__":
    main()