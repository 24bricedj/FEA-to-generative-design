"""Adjoint compliance sensitivity, validated against finite differences.

The analytical sensitivity (self-adjoint compliance) is the result derived by
hand in Phase 4:

    dc/drho_j = - u_j^T K_j^(0) u_j

The finite-difference check nudges each density, re-solves, and measures the
actual change in compliance -- an independent brute-force reference to validate
the analytical gradient against.
"""
import numpy as np
from solver import (build_mesh, element_stiffness, element_dofs,
                    assemble_stiffness, apply_bc, solve, reconstruct)


def compliance(nodes, conn, rho, E, nu, t, F, fixed):
    """Solve the FEA problem and return (compliance, displacement).
    Compliance c = F^T u."""
    K = assemble_stiffness(nodes, conn, rho, E, nu, t)
    Kr, Fr, free = apply_bc(K, F, fixed)
    ur = solve(Kr, Fr)
    u = reconstruct(ur, free, 2*len(nodes))
    c = F @ u
    return c, u


def analytical_sensitivity(nodes, conn, rho, E, nu, t, u):
    """Compliance sensitivity via the adjoint method.

    YOUR JOB — implement the result you derived:

        dc/drho_j = - u_j^T K_j^(0) u_j     for every element j

    where:
      - u_j       is the displacement at element j's 6 DOFs
      - K_j^(0)   is element j's full-material stiffness matrix

    Steps:
      1. make an array dc of length len(conn) (one sensitivity per element)
      2. loop over each element (use enumerate(conn) to get index ei and el)
      3. for each: get its full-material stiffness k0 = element_stiffness(...)
                   get its element displacement ue = u[element_dofs(el)]
                   compute dc[ei] = -(ue @ k0 @ ue)
      4. return dc

    Notes:
      - element_stiffness(coords, E, nu, t) wants the node COORDINATES:
        pass nodes[el].
      - ue @ k0 @ ue in NumPy computes the scalar u_j^T K_j^(0) u_j directly.
    """

    dc = np.zeros(len(conn))
    for ei, el in enumerate(conn):
        k0 = element_stiffness(nodes[el], E, nu, t)
        ue = u[element_dofs(el)]
        dc[ei] = -(ue @ k0 @ ue)
    return dc


def finite_diff_sensitivity(nodes, conn, rho, E, nu, t, F, fixed, eps=1e-6):
    """Brute-force reference: nudge each rho_j by eps, re-solve, measure
    dc/drho_j = (c(rho_j + eps) - c(rho_j)) / eps."""
    c0, _ = compliance(nodes, conn, rho, E, nu, t, F, fixed)
    dc = np.zeros(len(conn))
    for j in range(len(conn)):
        rp = rho.copy(); rp[j] += eps
        cj, _ = compliance(nodes, conn, rp, E, nu, t, F, fixed)
        dc[j] = (cj - c0) / eps
    return dc


def main():
    E, nu, t = 1.0, 0.3, 1.0
    nodes, conn = build_mesh(4.0, 2.0, 4, 2)   # small mesh: 16 elements
    ndof = 2*len(nodes)
    np.random.seed(0)
    rho = 0.5 + 0.4*np.random.rand(len(conn))  # random densities in [0.5, 0.9]

    # cantilever: fix left edge, downward load near bottom-right corner
    fixed = []
    for idx, (x, y) in enumerate(nodes):
        if abs(x) < 1e-9:
            fixed += [2*idx, 2*idx+1]
    F = np.zeros(ndof)
    tip = np.argmin((nodes[:, 0]-4.0)**2 + (nodes[:, 1]-0.0)**2)
    F[2*tip+1] = -1.0

    c, u = compliance(nodes, conn, rho, E, nu, t, F, fixed)
    dc_an = analytical_sensitivity(nodes, conn, rho, E, nu, t, u)
    dc_fd = finite_diff_sensitivity(nodes, conn, rho, E, nu, t, F, fixed)

    print(f"Compliance: {c:.6e}\n")
    print(f"{'elem':>4} {'analytical':>15} {'finite-diff':>15} {'rel err':>12}")
    print("-"*50)
    for j in range(len(conn)):
        rel = abs(dc_an[j]-dc_fd[j]) / (abs(dc_fd[j]) + 1e-30)
        print(f"{j:>4} {dc_an[j]:>15.6e} {dc_fd[j]:>15.6e} {rel:>12.2e}")
    maxrel = np.max(np.abs(dc_an-dc_fd) / (np.abs(dc_fd) + 1e-30))
    print(f"\nMax relative error: {maxrel:.2e}")


if __name__ == "__main__":
    main()