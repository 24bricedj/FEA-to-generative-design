"""
3D SIMP on the simplified engine bracket -- the first part with real
supports, a real distributed load, and non-design regions.

Usage:
    python mesh_bracket3d.py 0.12     # -> bracket.msh
    python optimize_bracket.py bracket.msh

Everything here is pieces you already built:
    surface_load            (fem3d_solve.py)  load spread over the pin hole
    oc_update(keep=...)     (optimize_mesh.py) rings locked solid, counted in budget
    the tet element, filter, sensitivities    unchanged from the 3D MBB
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import meshio

from mesh_io import read_mesh_3d, read_element_sets
from fem3d_solve import precompute_3d, fea_solve_3d, surface_load
from fem_mesh import build_filter, apply_filter, compliance_and_sensitivity
from optimize_mesh import oc_update, move_schedule

YP = 1.1          # pin-hole center height (must match mesh_bracket3d.py)

def pin_load(nodes, pin_faces, ndof, d, T, axis_pt):
    d = np.asarray(d, dtype=float)
    assert abs(d[2]) < 1e-12, "pin load must be perpendicular to the pin (z)"
    d = d/np.linalg.norm(d)       # make d a unit vector, so T is the true total
    """Load T on the pin hole, pulling in unit direction d (perpendicular
    to the pin axis). Only the half of the hole facing d bears."""
    cent = nodes[pin_faces].mean(axis=1)
    r = cent - axis_pt
    r[:, 2] = 0
    tri = pin_faces[r @ d > 0]
    F = surface_load(nodes, tri, ndof, direction=0, total=T*d[0]) \
      + surface_load(nodes, tri, ndof, direction=1, total=T*d[1])
    return F

def main(fname, volfrac=0.25, penal=3.0, rmin=0.25, nit=300):
    t0 = time.time()
    nodes, tets, sets, faces = read_mesh_3d(fname)
    esets = read_element_sets(fname)
    KE, V, edof, cent = precompute_3d(nodes, tets)
    ndof = 3*len(nodes)
    keep = esets["keep"]
    void = esets["void"]
    print(f"mesh: {len(nodes)} nodes, {len(tets)} tets, {ndof} DOFs; "
          f"keep = {V[keep].sum()/V.sum():.1%}, void = {V[void].sum()/V.sum():.1%} "
          f"of the volume  "
          f"(setup {time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # YOUR JOB: three short pieces, all things you just derived.
    #
    # 1. SUPPORTS. sets["bolt"] is the array of node numbers on the bolt-hole
    #    walls. Model each bolt as a rigid pin: fix ALL THREE DOFs of every
    #    one of those nodes. Build `fixed`, an array of DOF numbers.
    #    (In the MBB you fixed one direction per set, e.g. 3*sets["sym_x"] + 0.
    #     Here you need +0, +1 and +2 -- np.concatenate([...]) joins arrays.)
    #
    # 2. LOAD. faces["pin"] is the (k, 3) array of triangles on the pin-hole
    #    wall. The clevis pin pulls UP (+y) with total force 1. A pin pulling
    #    up only presses on the UPPER half of the hole -- the lower half
    #    separates from it. So:
    #      - compute each triangle's centroid height:
    #            yc = nodes[faces["pin"]][:, :, 1].mean(axis=1)
    #      - keep only the triangles with yc > YP   (boolean mask: arr[yc > YP])
    #      - call surface_load on those, direction=1, total=+1.0
    #
    # 3. START VALUE. rho starts at volfrac everywhere -- then lock the keep
    #    elements so the very first FEA solve already sees them solid.
    #
    # 4. (NEW) VOID. esets["void"] is now loaded above as `void`. Lock those
    #    elements at the minimum density at the start too -- and give your
    #    oc_update a `void=None` parameter that does the same every iteration
    #    (same spot and same guard as keep).
    # ------------------------------------------------------------------
    b = sets["bolt"]
    fixed = np.concatenate([3*b, 3*b+1, 3*b+2])
    yc = nodes[faces["pin"]][:, :, 1].mean(axis=1)
    top = faces["pin"][yc > YP]
    F = surface_load(nodes, top, ndof, direction=1, total=1.0)
    F2 = pin_load(nodes, faces["pin"], ndof, d=(0, 1, 0), T=1.0, axis_pt=np.array([2.0, YP, 0.0]))
    print("pin_load matches old load:", np.allclose(F, F2))
    rho = volfrac*np.ones(len(tets))
    rho[keep] = 1.0
    # piece 4 (void) goes here

    print(f"supports: {len(fixed)} fixed DOFs   load: total Fy = {F[1::3].sum():+.3f}")

    H, Hs = build_filter(cent, rmin)
    c_hist = []
    for it in range(nit):
        u = fea_solve_3d(rho, penal, KE, edof, ndof, fixed, F)
        c, dc = compliance_and_sensitivity(rho, penal, KE, edof, u)
        dc = apply_filter(H, Hs, rho, dc, V)
        rho = oc_update(rho, V, volfrac, dc, move_schedule(it),
                         keep=keep, void=void)

        c_hist.append(c)
        if it % 10 == 0:
            print(f"  it {it:3d}  c={c:9.3f}  vol={(rho*V).sum()/V.sum():.3f}  "
                  f"keep min={rho[keep].min():.2f}  void max={rho[void].max():.3f}  "
                  f"({time.time()-t0:.0f}s)")
        if it > 80 and (max(c_hist[-20:]) - min(c_hist[-20:]))/c < 1e-3:
            print(f"  converged at it={it}")
            break

    vol = (rho*V).sum()/V.sum()
    grey = np.sum(V*4*rho*(1 - rho))/V.sum()
    print(f"final: c={c:.3f}  vol={vol:.3f}  greyness={grey:.3f}  "
          f"({time.time()-t0:.0f}s total)")

    stem = fname.rsplit(".", 1)[0]
    region = np.zeros(len(tets))      # 0 design, 1 keep, -1 void (for ParaView)
    region[keep] = 1.0
    region[void] = -1.0
    meshio.write(f"{stem}_result.vtu",
                 meshio.Mesh(nodes, [("tetra", tets)],
                             cell_data={"rho": [rho],
                                        "region": [region]}))

    # quick look: solid elements from two angles (ParaView is the real view)
    s = rho > 0.5
    L, Hh, W = nodes.max(axis=0)
    fig = plt.figure(figsize=(14, 6))
    for k, (elev, azim) in enumerate([(20, -60), (20, 30)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(cent[s, 0], cent[s, 2], cent[s, 1], c=cent[s, 1],
                   cmap="viridis", s=2, depthshade=True)
        ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y")
        ax.set_box_aspect((L, W, Hh))
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle(f"bracket, {len(tets)} tets, volfrac {volfrac}, c = {c:.2f}")
    fig.tight_layout()
    fig.savefig(f"{stem}_result.png", dpi=150, bbox_inches="tight")
    print(f"saved {stem}_result.png and {stem}_result.vtu")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bracket.msh")