"""
Pure thermal curling demo: a cantilever bar clamped at the LEFT end, with a
through-depth temperature gradient (hot bottom, cool top) and NO mechanical
load. The hot bottom expands more than the cool top, so the bar curls upward
like a bimetallic strip -- deformation driven entirely by thermal strain.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from thermal_load import assemble_thermal_load
from solver import (
    build_mesh, assemble_stiffness, apply_dirichlet_bc,
    solve_system, reconstruct_full_solution, recover_element_stress,
)


def assemble_conductivity(nodes, conn, k):
    n = len(nodes); K = np.zeros((n, n))
    for el in conn:
        A_, B_, C_ = el[0], el[1], el[2]
        coords = nodes[[A_, B_, C_]]
        x1, y1 = coords[0]; x2, y2 = coords[1]; x3, y3 = coords[2]
        detC = x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2); A = detC/2.0
        b1, b2, b3 = y2-y3, y3-y1, y1-y2
        c1, c2, c3 = x3-x2, x1-x3, x2-x1
        Bt = (1.0/(2.0*A))*np.array([[b1, b2, b3], [c1, c2, c3]])
        K[np.ix_([A_, B_, C_], [A_, B_, C_])] += Bt.T @ Bt * k * A
    return K


def thermal_dirichlet(K, F, fixed):
    n = K.shape[0]; fn = list(fixed.keys())
    free = np.setdiff1d(np.arange(n), fn); Fm = F.copy()
    for j, Tb in fixed.items():
        Fm -= K[:, j]*Tb
    return K[np.ix_(free, free)], Fm[free], free


def main():
    E = 69000.0; nu = 0.33; alpha = 23e-6; t = 10.0; k = 1.0
    L = 200.0; H = 20.0; nx, ny = 80, 8
    T_ref = 20.0      # stress-free at 20 C
    T_hot = 170.0     # bottom edge
    T_cool = 20.0     # top edge
    tol = 1e-9

    nodes, conn = build_mesh(L, H, nx, ny)
    n = len(nodes); ndof = 2*n

    # thermal solve: gradient hot bottom -> cool top
    Kt = assemble_conductivity(nodes, conn, k); Ft = np.zeros(n)
    tfixed = {}
    for idx, (x, y) in enumerate(nodes):
        if abs(y) < tol: tfixed[idx] = T_hot
        elif abs(y-H) < tol: tfixed[idx] = T_cool
    Ktr, Ftr, tfree = thermal_dirichlet(Kt, Ft, tfixed)
    Tr = np.linalg.solve(Ktr, Ftr); T = np.zeros(n); T[tfree] = Tr
    for j, Tb in tfixed.items(): T[j] = Tb

    # structural: ONLY thermal load, clamp LEFT end only -> free to curl
    K = assemble_stiffness(nodes, conn, E, nu, t)
    F_th = assemble_thermal_load(nodes, conn, T, T_ref, E, nu, alpha, t)
    fixed = []
    for idx, (x, y) in enumerate(nodes):
        if abs(x) < tol: fixed += [2*idx, 2*idx+1]
    Kr, Fr, free = apply_dirichlet_bc(K, F_th, fixed)
    ur = solve_system(Kr, Fr); u = reconstruct_full_solution(ur, free, ndof)

    # von Mises per element (thermal-corrected)
    vm = np.zeros(len(conn))
    for ei, el in enumerate(conn):
        coords = nodes[el]
        ue = np.array([u[2*el[0]], u[2*el[0]+1], u[2*el[1]],
                       u[2*el[1]+1], u[2*el[2]], u[2*el[2]+1]])
        sx, sy, txy = recover_element_stress(coords, ue, E, nu, alpha,
                                             T[el].mean()-T_ref)
        vm[ei] = np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)

    v = u.reshape(-1, 2)[:, 1]
    tipnodes = np.where(np.abs(nodes[:, 0]-L) < tol)[0]
    tip = np.mean([u[2*nd+1] for nd in tipnodes])
    print(f"Thermal curl: gradient {T_hot}->{T_cool} (ref {T_ref}), clamped left, NO mech load")
    print(f"  tip vertical displacement: {tip:.4f} mm  (curls {'up' if tip>0 else 'down'})")
    print(f"  max von Mises: {vm.max():.2f} MPa")

    # plot: deformed shape (true scale) colored by temperature
    scale = 1.0
    deformed = nodes + scale * u.reshape(-1, 2)
    Tel = np.array([T[el].mean() for el in conn])
    triang = mtri.Triangulation(deformed[:, 0], deformed[:, 1], conn)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    tpc = ax.tripcolor(triang, facecolors=Tel, cmap="inferno",
                       edgecolors="k", linewidth=0.1)
    ax.set_aspect("equal"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(f"Pure thermal curling: hot-bottom/cool-top gradient, "
                 f"clamped left, no mechanical load (scale={scale}x)")
    fig.colorbar(tpc, ax=ax, label="Temperature [C]")
    fig.savefig("thermal_curl.png", dpi=150, bbox_inches="tight")
    print("saved thermal_curl.png")


if __name__ == "__main__":
    main()