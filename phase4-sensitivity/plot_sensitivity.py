"""
Visualize the compliance sensitivity field: color each element by |dc/drho|,
so the regions where material matters most for stiffness light up. On a solid
cantilever this reveals the load path -- exactly the map topology optimization
follows to decide where to keep material.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from solver import (build_mesh, assemble_stiffness, apply_bc, solve,
                    reconstruct)
from sensitivity import analytical_sensitivity


def main():
    E, nu, t = 1.0, 0.3, 1.0
    nodes, conn = build_mesh(8.0, 4.0, 32, 16)
    ndof = 2*len(nodes)

    rho = np.ones(len(conn))   # solid start: all densities = 1

    # cantilever: clamp left edge, downward load at mid-right
    fixed = []
    for idx, (x, y) in enumerate(nodes):
        if abs(x) < 1e-9:
            fixed += [2*idx, 2*idx+1]
    F = np.zeros(ndof)
    tip = np.argmin((nodes[:, 0]-8.0)**2 + (nodes[:, 1]-2.0)**2)
    F[2*tip+1] = -1.0

    K = assemble_stiffness(nodes, conn, rho, E, nu, t)
    Kr, Fr, free = apply_bc(K, F, fixed)
    ur = solve(Kr, Fr)
    u = reconstruct(ur, free, ndof)

    dc = analytical_sensitivity(nodes, conn, rho, E, nu, t, u)
    sens = np.abs(dc)   # magnitude: how much material here matters

    patches = [Polygon(nodes[el], closed=True) for el in conn]
    pc = PatchCollection(patches, cmap="inferno")
    pc.set_array(sens)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Compliance sensitivity field |dc/dρ| — where material matters most")
    fig.colorbar(pc, ax=ax, label="|dc/dρ|  (element strain energy)")
    ax.plot(nodes[tip, 0], nodes[tip, 1], "co", ms=8, mec="k", label="load")
    ax.legend(loc="upper right")
    fig.savefig("sensitivity_field.png", dpi=150, bbox_inches="tight")
    print(f"max sensitivity: {sens.max():.4f}  min: {sens.min():.4e}")
    print("saved sensitivity_field.png")


if __name__ == "__main__":
    main()