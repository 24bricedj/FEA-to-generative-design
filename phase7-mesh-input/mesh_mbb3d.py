"""
3D MBB beam as a thin slab, meshed in tetrahedra with Gmsh.

Two symmetry planes, so only a quarter of the full slab is modeled:
  x = 0  : midspan symmetry  (the 2D MBB symmetry edge, extruded)
  z = 0  : mid-thickness symmetry (front and back faces behave the same)

Physical groups:
  "domain"   (3D)  design volume
  "sym_x"    (2D)  face x = 0          -> ux fixed
  "sym_z"    (2D)  face z = 0          -> uz fixed
  "roller"   (1D)  edge x = W, y = 0   -> uy fixed  (2D roller point, extruded)
  "load"     (1D)  edge x = 0, y = H   -> line load in -y (2D load point, extruded)
"""
import sys
import gmsh


def make_mbb3d(W=3.0, H=1.0, T=0.1, h=0.05, fname="mbb3d.msh"):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("mbb3d")
    box = gmsh.model.occ.addBox(0, 0, 0, W, H, T)
    gmsh.model.occ.synchronize()

    tol = 1e-6
    def surf_at(**kw):
        return [s for d, s in gmsh.model.getEntities(2)
                if _on(gmsh.model.getBoundingBox(2, s), kw, tol)]
    def curve_at(**kw):
        return [c for d, c in gmsh.model.getEntities(1)
                if _on(gmsh.model.getBoundingBox(1, c), kw, tol)]

    gmsh.model.addPhysicalGroup(3, [box], name="domain")
    gmsh.model.addPhysicalGroup(2, surf_at(x=0), name="sym_x")
    gmsh.model.addPhysicalGroup(2, surf_at(z=0), name="sym_z")
    gmsh.model.addPhysicalGroup(1, curve_at(x=W, y=0), name="roller")
    gmsh.model.addPhysicalGroup(1, curve_at(x=0, y=H), name="load")

    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.model.mesh.generate(3)
    gmsh.write(fname)
    gmsh.finalize()
    print(f"wrote {fname}")


def _on(bb, kw, tol):
    """True if the entity's bounding box lies entirely on the given planes."""
    xmin, ymin, zmin, xmax, ymax, zmax = bb
    lo = {"x": xmin, "y": ymin, "z": zmin}
    hi = {"x": xmax, "y": ymax, "z": zmax}
    return all(abs(lo[a]-v) < tol and abs(hi[a]-v) < tol for a, v in kw.items())


if __name__ == "__main__":
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    make_mbb3d(h=h)