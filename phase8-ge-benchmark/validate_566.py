"""
Validate YOUR solver against MIT's OptiStruct results on the winning GE bracket
(SimJEB design 566, M Arie Kurniawan).

    python validate_566.py            (expects data/566.vtk and data/566.csv)

Frame (SimJEB): mm, N, MPa.  Pin runs along y (lugs at y ~ -89 and y ~ -60),
vertical is +z, "horizontal out" is -x.  In the CSV, surf == 2 marks the bolt
nodes and surf == 3 the pin-hole nodes.
"""
import time
import numpy as np
import pandas as pd
import meshio

from fem_fast import precompute_3d_fast
from fast_solve import FastSolver

LBF = 4.4482216          # newtons per pound-force
INCH = 25.4              # mm per inch
E, NU = 113.8e3, 0.342   # Ti-6Al-4V in MPa (N/mm^2), as SimJEB used


def rbe3_force(nodes, pin, ndof, vec):
    """A force vec = (Fx, Fy, Fz) spread over the pin nodes the RBE3 way.

    YOUR JOB: every pin node gets an EQUAL share of the force -- not area-
    weighted, not just the bearing half: all len(pin) nodes, both lugs.
    For each direction k = 0, 1, 2, add vec[k]/len(pin) into F[3*pin + k].
    (3*pin + k is an ARRAY of DOF numbers, so one line per k does every node.)
    """
    F = np.zeros(ndof)
    for k in range(3):
        F[3*pin + k] += vec[k]/len(pin)
    return F


def rbe3_moment_z(nodes, pin, ndof, Mz):
    """A torque Mz about the vertical (z) axis, spread over the pin nodes.

    YOUR JOB -- the bolt-group torsion formula from machine design:
        each node gets a force PERPENDICULAR to its arm r (measured from the
        group's center, in the x-y plane), with size proportional to |r|:
            f_i = Mz * (z_hat x r_i) / sum_j |r_j|^2
        z_hat x r = (-r_y, r_x, 0)  -- your rotation-about-z row!
    Steps:
      1. center = nodes[pin].mean(axis=0);  r = nodes[pin] - center
      2. den = sum over pin nodes of (r_x^2 + r_y^2)
      3. add  -Mz*r_y/den  into the x-DOFs and  Mz*r_x/den  into the y-DOFs
    """
    F = np.zeros(ndof)
    center = nodes[pin].mean(axis=0)
    r = nodes[pin] - center
    den = np.sum(r[:, 0]**2 + r[:, 1]**2)
    F[3*pin] += -Mz*r[:, 1]/den
    F[3*pin + 1] += Mz*r[:, 0]/den
    return F                     #    y-forces:   Mz * r_x / den


def main():
    t0 = time.time()
    mesh = meshio.read("data/566.vtk")
    nodes, tets = mesh.points, mesh.cells_dict["tetra"]
    ref = pd.read_csv("data/566.csv")          # MIT's results, one row per node
    ndof = 3*len(nodes)
    KE, V, edof, tets = precompute_3d_fast(nodes, tets, E, NU)
    print(f"mesh: {len(nodes)} nodes, {len(tets)} tets  (setup {time.time()-t0:.0f}s)")

    bolt = np.where(ref.surf.values == 2)[0]
    pin = np.where(ref.surf.values == 3)[0]
    fixed = np.concatenate([3*bolt, 3*bolt + 1, 3*bolt + 2])
    th = np.radians(42)
    cases = {
        "ver": rbe3_force(nodes, pin, ndof, (0, 0, 8000*LBF)),
        "hor": rbe3_force(nodes, pin, ndof, (-8500*LBF, 0, 0)),
        "dia": rbe3_force(nodes, pin, ndof, (-9500*LBF*np.sin(th), 0, 9500*LBF*np.cos(th))),
        "tor": rbe3_moment_z(nodes, pin, ndof, 5000*LBF*INCH),
    }

    names = list(cases)
    Fs = np.column_stack([cases[n] for n in names])
    solver = FastSolver(KE, edof, ndof, fixed, nodes)
    U = solver.solve(np.ones(len(tets)), 1.0, Fs)     # rho = 1: a real solid part
    print(f"solved 4 cases (CG steps {solver.iters})  ({time.time()-t0:.0f}s)\n")

    print(f"{'case':5s} {'max|u| yours':>13s} {'max|u| MIT':>11s} {'field error':>12s} {'compliance (N mm)':>18s}")
    for i, n in enumerate(names):
        u = U[:, i].reshape(-1, 3)
        u_ref = ref[[f"{n}_xdisp", f"{n}_ydisp", f"{n}_zdisp"]].values
        err = np.linalg.norm(u - u_ref)/np.linalg.norm(u_ref)
        print(f"{n:5s} {np.linalg.norm(u, axis=1).max():13.4f} "
              f"{np.linalg.norm(u_ref, axis=1).max():11.4f} {err:11.2%} "
              f"{Fs[:, i] @ U[:, i]:18.1f}")


if __name__ == "__main__":
    main()