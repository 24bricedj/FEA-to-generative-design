"""
The L-bracket: the standard benchmark for stress-constrained topology
optimization (e.g. Le et al. 2010, Holmberg et al. 2013).

    python mesh_lbracket.py 1.0       # element size in mm -> lbracket.msh

A 100 x 100 mm square with the top-right 60 x 60 mm removed, 1 mm thick.
The inside (re-entrant) corner at (40, 40) is sharp, so a stiffness-optimized
design puts material right where stress concentrates. A stress-constrained
design has to round that corner off.

Physical groups:  "domain"  |  "fixed" (top edge, y = 100)  |
                  "load" (right edge, the 8 mm just below the top of the arm)
"""
import sys
import gmsh


def main(h=1.0, fname="lbracket.msh"):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    g = gmsh.model.geo
    pts = [(0, 0), (100, 0), (100, 32), (100, 40), (40, 40), (40, 100), (0, 100)]
    p = [g.addPoint(x, y, 0) for x, y in pts]
    lines = [g.addLine(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
    loop = g.addCurveLoop(lines)
    surf = g.addPlaneSurface([loop])
    g.synchronize()
    gmsh.model.addPhysicalGroup(2, [surf], name="domain")
    gmsh.model.addPhysicalGroup(1, [lines[5]], name="fixed")      # (40,100)-(0,100)
    gmsh.model.addPhysicalGroup(1, [lines[2]], name="load")       # (100,32)-(100,40)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.model.mesh.generate(2)
    gmsh.write(fname)
    n = len(gmsh.model.mesh.getElementsByType(2)[0])
    gmsh.finalize()
    print(f"wrote {fname}: {n} triangles")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)