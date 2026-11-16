"""
Turn the optimized density field into a real, smooth part, then score it
properly against the GE winner.

    python optimize_ge.py                # -> ge_result.vtu   (do this first)
    python rescore_smooth.py [h]         # h = re-mesh size in mm (default 1.0)

Pipeline
  1. element density -> nodal density (volume-weighted average; keep nodes = 1)
  2. sample that density onto a fine voxel grid (0.75 mm); outside the envelope
     (holes, clevis gap, air) the density is 0
  3. marching cubes: the surface where density = t, with t chosen by bisection so
     the enclosed volume is EXACTLY the winner's. Zero outside the envelope means
     the surface always closes -- a watertight solid (the ParaView Clip lesson)
  4. Taubin smoothing (shrink step + inflate step: smooth without losing volume),
     then decimate to a sensible triangle count
  5. write ge_smooth.stl  -- a printable file
  6. fill the STL with fine, good-quality tets (TetGen)
  7. re-score: same RBE3 loads, same supports, same validated solver; compliance
     per case + peak von Mises (nodal-averaged, like SimJEB's values)

Needs:  pip install pyvista tetgen
"""
import sys
import time
import numpy as np
import meshio
import pyvista as pv
import tetgen
from scipy.spatial import cKDTree

from fem_fast import precompute_3d_fast            # also puts phase7 on the path
from fast_solve import FastSolver
from validate_566 import rbe3_force, rbe3_moment_z, LBF, INCH, E, NU
from optimize_ge import von_mises, WINNER_VOLUME, WINNER, WINNER_STRESS

# envelope geometry (Dolfiny's frame), for re-finding the interfaces after re-meshing
PIN_Y, PIN_Z, PIN_R = -56.85, 44.62, 9.55
LUG_X = [(-63.7, -55.8), (-92.2, -84.4)]
BOLTS = [(-148.06, -33.61), (-148.06, 18.47), (0.0, -38.11), (0.0, 0.0)]   # (x, y)
BOLT_R, BOSS_R, BOSS_TOP = 5.2, 8.0, 7.95


# ---------------------------------------------------------------- 2-4. surface
def voxel_density(grid, h_vox):
    """Sample the nodal density onto a regular grid; 0 outside the envelope."""
    lo = np.array(grid.bounds[::2]) - 2*h_vox
    hi = np.array(grid.bounds[1::2]) + 2*h_vox
    dims = np.ceil((hi - lo)/h_vox).astype(int) + 1
    img = pv.ImageData(dimensions=dims, spacing=(h_vox,)*3, origin=lo)
    smp = img.sample(grid)
    img.point_data["rho"] = np.where(smp["vtkValidPointMask"] > 0, smp["rho"], 0.0)
    return img


def surface_at_volume(img, target):
    """Bisection on the iso-level t until the closed surface encloses `target`."""
    lo, hi = 0.02, 0.98
    for _ in range(40):
        t = 0.5*(lo + hi)
        surf = img.contour([t], scalars="rho", method="flying_edges")
        if surf.volume > target:
            lo = t                        # too much material: raise the level
        else:
            hi = t
        if abs(surf.volume - target)/target < 2e-4:
            break
    return surf, t


# ---------------------------------------------------------------- 6. re-mesh
def remesh(surf, h):
    """Fill the closed surface with quality tets (TetGen). maxvolume ~ h^3/5
    gives tets about h on a side; minratio/mindihedral reject slivers."""
    tg = tetgen.TetGen(surf.triangulate())
    tg.tetrahedralize(order=1, mindihedral=10, minratio=1.5, maxvolume=h**3/5, quality=True)
    g = tg.grid
    return np.asarray(g.points), g.cells_dict[pv.CellType.TETRA]


# ---------------------------------------------------------------- 7. interfaces
def boundary_nodes(tets):
    """Nodes on the outside of a tet mesh (faces that belong to only one tet)."""
    f = np.vstack([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                   tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    fs = np.sort(f, axis=1)
    _, idx, cnt = np.unique(fs, axis=0, return_index=True, return_counts=True)
    return np.unique(fs[idx[cnt == 1]])


def find_interfaces(nodes, tets):
    b = boundary_nodes(tets)
    p = nodes[b]
    # pin: on the hole wall (radius PIN_R from the pin axis) inside a lug
    rp = np.hypot(p[:, 1] - PIN_Y, p[:, 2] - PIN_Z)
    in_lug = np.zeros(len(b), bool)
    for x0, x1 in LUG_X:
        in_lug |= (p[:, 0] > x0) & (p[:, 0] < x1)
    pin = b[(np.abs(rp - PIN_R) < 0.6) & in_lug]
    # bolts: hole wall, washer face and seat of each boss (not its outer wall)
    bolt = []
    for bx, by in BOLTS:
        rb = np.hypot(p[:, 0] - bx, p[:, 1] - by)
        bolt.append(b[(rb < BOSS_R - 0.4) & (p[:, 2] < BOSS_TOP + 0.3)])
    return np.concatenate(bolt), pin


def nodal_average(nodes, tets, V, field):
    num = np.zeros(len(nodes))
    den = np.zeros(len(nodes))
    np.add.at(num, tets.ravel(), np.repeat(field*V, 4))
    np.add.at(den, tets.ravel(), np.repeat(V, 4))
    return num/np.maximum(den, 1e-30)


def main(h=1.0):
    t0 = time.time()
    # ---- 1. nodal density
    res = meshio.read("ge_result.vtu")
    nodes, tets = res.points, res.cells_dict["tetra"]
    rho, keep = res.cell_data["rho"][0], res.cell_data["keep"][0] > 0.5
    C = np.concatenate([np.ones((len(tets), 4, 1)), nodes[tets]], axis=2)
    V = np.abs(np.linalg.det(C))/6
    rn = nodal_average(nodes, tets, V, rho)
    rn[np.unique(tets[keep])] = 1.0                    # keep regions fully solid
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: tets}, nodes)
    grid.point_data["rho"] = rn

    # ---- 2-3. closed surface at the winner's volume
    img = voxel_density(grid, 0.75)
    surf, t = surface_at_volume(img, WINNER_VOLUME)
    print(f"marching cubes at rho = {t:.3f}: {surf.n_cells} triangles, "
          f"volume {surf.volume:.0f} mm^3, open edges = {surf.n_open_edges} (0 = watertight)")

    # ---- 4. smooth (volume-preserving), then thin out the triangles
    surf = surf.smooth_taubin(n_iter=40, pass_band=0.05)
    surf = surf.decimate(0.6, volume_preservation=True).clean()
    surf = surf.compute_normals(consistent_normals=True, auto_orient_normals=True,
                                cell_normals=True, point_normals=False)
    print(f"smoothed + decimated: {surf.n_cells} triangles, volume {surf.volume:.0f} mm^3, "
          f"open edges = {surf.n_open_edges}")

    # ---- 5. the printable file
    surf.save("ge_smooth.stl")
    print(f"saved ge_smooth.stl  ({time.time()-t0:.0f}s)")

    # ---- 6. fine re-mesh
    nodes, tets = remesh(surf, h)
    KE, V, edof, tets = precompute_3d_fast(nodes, tets, E, NU)
    ndof = 3*len(nodes)
    print(f"re-mesh (h = {h} mm): {len(tets)} tets, {len(nodes)} nodes, "
          f"volume {V.sum():.0f} mm^3 = {V.sum()*4.47e-3:.0f} g  ({time.time()-t0:.0f}s)")

    # ---- 7. re-score, exactly as in the validation
    bolt, pin = find_interfaces(nodes, tets)
    print(f"interfaces found: {len(bolt)} bolt nodes, {len(pin)} pin nodes")
    fixed_dofs = np.concatenate([3*bolt, 3*bolt + 1, 3*bolt + 2])
    th = np.radians(42)
    cases = {
        "ver": rbe3_force(nodes, pin, ndof, (0, 0, 8000*LBF)),
        "hor": rbe3_force(nodes, pin, ndof, (0, -8500*LBF, 0)),
        "dia": rbe3_force(nodes, pin, ndof, (0, -9500*LBF*np.sin(th), 9500*LBF*np.cos(th))),
        "tor": rbe3_moment_z(nodes, pin, ndof, 5000*LBF*INCH),
    }
    names = list(cases)
    Fs = np.column_stack([cases[n] for n in names])
    solver = FastSolver(KE, edof, ndof, fixed_dofs, nodes)
    U = solver.solve(np.ones(len(tets)), 1.0, Fs)

    # stress right AT a fully-fixed patch edge is a modeling artifact (a rigid
    # support makes a theoretical singularity there), so also report the peak
    # more than 3 mm from any bolt or pin node -- the winner's peaks all sit there
    d_if = np.minimum(cKDTree(nodes[bolt]).query(nodes)[0], cKDTree(nodes[pin]).query(nodes)[0])
    away = d_if > 3.0
    print(f"\n{'case':5s} {'yours (N mm)':>14s} {'winner (N mm)':>14s} {'ratio':>7s} "
          f"{'peak vM':>9s} {'peak vM >3mm from supports/pin':>32s}")
    peak_all, peak_away, stress = 0.0, 0.0, {}
    for i, n in enumerate(names):
        ci = Fs[:, i] @ U[:, i]
        vm = nodal_average(nodes, tets, V, von_mises(nodes, tets, U[:, i], E, NU))
        stress[n] = vm
        peak_all = max(peak_all, vm.max())
        peak_away = max(peak_away, vm[away].max())
        print(f"{n:5s} {ci:14.1f} {WINNER[n]:14.1f} {ci/WINNER[n]:6.2f}x "
              f"{vm.max():9.0f} {vm[away].max():32.0f}")
    print(f"\npeak von Mises, all nodes:            {peak_all:.0f} MPa  (dominated by support-edge artifacts)")
    print(f"peak von Mises, >3 mm from interfaces: {peak_away:.0f} MPa  "
          f"<- compare: winner {WINNER_STRESS:.0f} MPa, yield 903 MPa")

    meshio.write("ge_smooth_result.vtu", meshio.Mesh(
        nodes, [("tetra", tets)],
        point_data={**{f"vm_{n}": stress[n] for n in names},
                    "dist_to_interface": d_if}))
    print(f"saved ge_smooth_result.vtu  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)