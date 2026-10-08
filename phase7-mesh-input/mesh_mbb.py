"""
Generate an UNSTRUCTURED triangle mesh of the MBB half-domain with Gmsh,
tagging boundaries as physical groups so the solver can find them by name.

Physical groups:
  "domain"    - the 2D surface (design region)
  "symmetry"  - left edge  (x fixed: symmetry plane at midspan)
  "roller"    - bottom-right corner point (y fixed)
  "load"      - top-left corner point (downward point load)
"""
import sys
import gmsh


def make_mbb_mesh(W=3.0, H=1.0, h=0.05, fname="mbb.msh"):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("mbb")

    # corner points: p1 bottom-left, p2 bottom-right, p3 top-right, p4 top-left
    p1 = gmsh.model.geo.addPoint(0, 0, 0, h)
    p2 = gmsh.model.geo.addPoint(W, 0, 0, h)
    p3 = gmsh.model.geo.addPoint(W, H, 0, h)
    p4 = gmsh.model.geo.addPoint(0, H, 0, h)

    l_bot = gmsh.model.geo.addLine(p1, p2)
    l_rgt = gmsh.model.geo.addLine(p2, p3)
    l_top = gmsh.model.geo.addLine(p3, p4)
    l_lft = gmsh.model.geo.addLine(p4, p1)

    loop = gmsh.model.geo.addCurveLoop([l_bot, l_rgt, l_top, l_lft])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    # physical groups: names the solver looks up
    gmsh.model.addPhysicalGroup(2, [surf], name="domain")
    gmsh.model.addPhysicalGroup(1, [l_lft], name="symmetry")
    gmsh.model.addPhysicalGroup(0, [p2], name="roller")
    gmsh.model.addPhysicalGroup(0, [p4], name="load")

    gmsh.model.mesh.generate(2)          # unstructured triangles
    gmsh.write(fname)
    gmsh.finalize()
    print(f"wrote {fname}")


if __name__ == "__main__":
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    make_mbb_mesh(h=h)