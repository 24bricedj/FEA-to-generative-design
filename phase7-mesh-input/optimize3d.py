"""
3D SIMP topology optimization on a tetrahedral mesh.

Usage:
    python mesh_mbb3d.py 0.05     # thin-slab MBB, quarter model -> mbb3d.msh
    python optimize3d.py mbb3d.msh

Reuses, unchanged:
    oc_update        (yours, from optimize_mesh.py -- V is now tet volume)
    build_filter     (KD-tree on centroids; works in 3D as-is)
    apply_filter     (per-unit-volume averaging)
    compliance_and_sensitivity  (einsum over any element size)
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio

from mesh_io import read_mesh_3d
from fem3d_solve import precompute_3d, fea_solve_3d, line_load
from fem_mesh import build_filter, apply_filter, compliance_and_sensitivity
from optimize_mesh import oc_update, move_schedule


def mirror(nodes, tets, rho, axis):
    """Reflect a tet mesh across the plane coordinate[axis] = 0 and append the
    copy. Mirroring flips each tet's orientation, so two nodes are swapped to
    keep every volume positive."""
    m = nodes.copy()
    m[:, axis] *= -1
    t2 = tets[:, [0, 2, 1, 3]] + len(nodes)
    allnodes = np.vstack([nodes, m])
    alltets = np.vstack([tets, t2])
    # merge the duplicate nodes on the mirror plane so the two halves are one
    # connected mesh (otherwise viewers see a seam and pull it apart)
    key = np.round(allnodes/1e-9).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True,
                                  return_inverse=True)
    return allnodes[first], inverse.ravel()[alltets], np.concatenate([rho, rho])


def main(fname, volfrac=0.5, penal=3.0, rmin=0.075, nit=300):
    t0 = time.time()
    nodes, tets, sets, segs = read_mesh_3d(fname)
    KE, V, edof, cent = precompute_3d(nodes, tets)
    ndof = 3*len(nodes)
    T = nodes[:, 2].max()              # modeled (half) thickness
    print(f"mesh: {len(nodes)} nodes, {len(tets)} tets, {ndof} DOFs, "
          f"thickness modeled {T:g}  (setup {time.time()-t0:.1f}s)")

    # boundary conditions from physical groups
    fixed = np.unique(np.concatenate([
        3*sets["sym_x"] + 0,           # x-symmetry face: ux = 0
        3*sets["sym_z"] + 2,           # z-symmetry face: uz = 0
        3*sets["roller"] + 1,          # roller edge:     uy = 0
    ]))
    # line load: 1 unit of force per unit thickness, matching the 2D problem
    F = line_load(nodes, segs["load"], ndof, direction=1, total=-T)

    H, Hs = build_filter(cent, rmin)
    rho = volfrac*np.ones(len(tets))
    c_hist = []

    for it in range(nit):
        u = fea_solve_3d(rho, penal, KE, edof, ndof, fixed, F)
        c, dc = compliance_and_sensitivity(rho, penal, KE, edof, u)
        dc = apply_filter(H, Hs, rho, dc, V)
        rho = oc_update(rho, V, volfrac, dc, move_schedule(it))

        c_hist.append(c)
        if it % 25 == 0:
            print(f"  it {it:3d}  c/T={c/T:8.2f}  "
                  f"vol={(rho*V).sum()/V.sum():.3f}  ({time.time()-t0:.0f}s)")
        if it > 80 and (max(c_hist[-20:]) - min(c_hist[-20:]))/c < 1e-3:
            print(f"  converged at it={it}")
            break

    vol = (rho*V).sum()/V.sum()
    print(f"final: c/T={c/T:.2f}  vol={vol:.3f}  ({time.time()-t0:.0f}s total)")
    print("  (c/T is compliance per unit thickness -- compare to the 2D "
          "unstructured result, 202.8)")

    stem = fname.rsplit(".", 1)[0]

    # 1. ParaView file: open it, Threshold rho > 0.5 to see the 3D part
    meshio.write(f"{stem}_result.vtu",
                 meshio.Mesh(nodes, [("tetra", tets)],
                             cell_data={"rho": [rho]}))
    # full part: undo both symmetry planes (x = 0 midspan, z = 0 mid-thickness)
    fn, ft, fr = mirror(nodes, tets, rho, axis=0)
    fn, ft, fr = mirror(fn, ft, fr, axis=2)
    meshio.write(f"{stem}_full.vtu",
                 meshio.Mesh(fn, [("tetra", ft)], cell_data={"rho": [fr]}))

    # 2. Quick-look figure
    #    top: front view -- density averaged through the thickness onto an
    #         x-y grid (volume-weighted), directly comparable to the 2D plots
    #    bottom: isometric view of the solid elements (rho > 0.5)
    W, Hh = nodes[:, 0].max(), nodes[:, 1].max()
    # bins about one element wide, so every bin catches some tets;
    # any bin that still catches none is filled from its nearest neighbor
    h_el = np.mean(V)**(1/3)*1.5
    nx, ny = max(int(W/h_el), 10), max(int(Hh/h_el), 4)
    ix = np.clip((cent[:, 0]/W*nx).astype(int), 0, nx - 1)
    iy = np.clip((cent[:, 1]/Hh*ny).astype(int), 0, ny - 1)
    num = np.zeros((ny, nx)); den = np.zeros((ny, nx))
    np.add.at(num, (iy, ix), rho*V)
    np.add.at(den, (iy, ix), V)
    front = np.where(den > 0, num/np.maximum(den, 1e-30), np.nan)
    if np.isnan(front).any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(np.isnan(front), return_distances=False,
                                     return_indices=True)
        front = front[tuple(idx)]

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    im = ax1.imshow(front, origin="lower", extent=[0, W, 0, Hh],
                    cmap="jet", vmin=0, vmax=1, interpolation="bilinear")
    ax1.set_title(f"3D MBB, front view (thickness-averaged), "
                  f"{len(tets)} tets, c/T = {c/T:.1f}")
    fig.colorbar(im, ax=ax1, label="density ρ")

    ax2 = fig.add_subplot(2, 1, 2, projection="3d")
    s = rho > 0.5
    ax2.scatter(cent[s, 0], cent[s, 2], cent[s, 1], c=cent[s, 1],
                cmap="viridis", s=3, depthshade=True)
    ax2.set_xlabel("x"); ax2.set_ylabel("z"); ax2.set_zlabel("y")
    ax2.set_box_aspect((W, 3*T, Hh))
    ax2.view_init(elev=20, azim=-60)
    ax2.set_title("solid elements (ρ > 0.5), isometric")
    fig.tight_layout()
    fig.savefig(f"{stem}_result.png", dpi=150, bbox_inches="tight")
    print(f"saved {stem}_result.png, {stem}_result.vtu, and {stem}_full.vtu (whole beam)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mbb3d.msh")