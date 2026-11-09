"""
Simplified engine-mount bracket (a "GE bracket lite"), meshed in tets.

Design domain: a block  L (x) by H (y) by W (z),  sitting on the plane y = 0.

  - 4 bolt holes through a base flange of thickness tb, at the corners.
    Bolted to the engine: these are the supports.
  - Above each bolt, a CLEARANCE column (bolt head + socket access) that
    must stay EMPTY ("void"), from the flange top to the top of the block.
  - 1 pin hole through the block (along z) near the top middle.
    A clevis pin pulls on it: this is where the load goes in.
  - A ring of material around every hole is NON-DESIGN ("keep"): it must
    stay solid so the bolts and pin have something to bear on.

        y
        |      ___________________________
        |     |            (O) pin        |
        |     |                           |
        |     |: :|                   |: :|   <- clearance columns (void)
        |     |_|_|___________________|_|_|   <- bolt holes through flange
        +-------------------------------------- x

Physical groups:
  "domain"  (3D)  every tet (design + keep)
  "keep"    (3D)  the rings around the holes   -> rho locked at 1
  "void"    (3D)  clearance columns            -> rho locked at the minimum
  "bolt"    (2D)  bolt-hole walls             -> supports
  "pin"     (2D)  pin-hole wall               -> load
"""
import sys
import gmsh


def make_bracket(L=4.0, H=1.5, W=2.0,
                 rb=0.15, tb=0.3, inset=0.4,      # bolt holes
                 rp=0.15, yp=1.1,                 # pin hole
                 ring=0.15,                       # keep-ring thickness
                 h=0.12, fname="bracket.msh"):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("bracket")
    occ = gmsh.model.occ

    box = occ.addBox(0, 0, 0, L, H, W)

    bolts = [(inset, inset), (L - inset, inset),
             (inset, W - inset), (L - inset, W - inset)]       # (x, z)
    holes, rings, columns = [], [], []
    for x, z in bolts:
        # vertical cylinders: start at y = 0, axis +y, through the flange
        holes.append(occ.addCylinder(x, 0, z, 0, tb, 0, rb))
        rings.append(occ.addCylinder(x, 0, z, 0, tb, 0, rb + ring))
        # clearance column: sits on the ring (the bolt head bears on the
        # ring's top face), same radius as the ring, up to the top of the block
        columns.append(occ.addCylinder(x, tb, z, 0, H - tb, 0, rb + ring))
    # pin: horizontal cylinder along z, through the whole width
    holes.append(occ.addCylinder(L/2, yp, 0, 0, 0, W, rp))
    rings.append(occ.addCylinder(L/2, yp, 0, 0, 0, W, rp + ring))

    # drill the holes out of the block AND out of the rings
    body, _ = occ.cut([(3, box)], [(3, t) for t in holes], removeTool=False)
    ring_vols, _ = occ.cut([(3, t) for t in rings], [(3, t) for t in holes])
    # fragment: split the block where the rings and columns are, so the mesh
    # conforms to their boundaries (every tet is fully design, keep, or void)
    tools = ring_vols + [(3, t) for t in columns]
    out, pieces = occ.fragment(body, tools)
    occ.synchronize()
    # pieces[i] = what input i became; inputs were [body] + rings + columns
    nr = len(ring_vols)
    keep = sorted({t for p in pieces[1:1 + nr] for d, t in p})
    void = sorted({t for p in pieces[1 + nr:] for d, t in p})
    vols = [t for d, t in gmsh.model.getEntities(3)]

    def surfaces_in_cylinder(x0, y0, z0, axis, r, length):
        """Surfaces lying inside a cylinder's bounding box -> the hole walls."""
        e = 1e-4
        lo = [x0 - r, y0 - r, z0 - r]
        hi = [x0 + r, y0 + r, z0 + r]
        lo[axis] = [x0, y0, z0][axis]
        hi[axis] = [x0, y0, z0][axis] + length
        found = gmsh.model.getEntitiesInBoundingBox(
            lo[0]-e, lo[1]-e, lo[2]-e, hi[0]+e, hi[1]+e, hi[2]+e, dim=2)
        # walls only: skip flat caps (zero extent along the hole axis)
        return [t for d, t in found
                if gmsh.model.getBoundingBox(2, t)[axis + 3]
                 - gmsh.model.getBoundingBox(2, t)[axis] > 1e-3]

    bolt_surfs = []
    for x, z in bolts:
        bolt_surfs += surfaces_in_cylinder(x, 0, z, 1, rb, tb)
    pin_surfs = surfaces_in_cylinder(L/2, yp, 0, 2, rp, W)

    gmsh.model.addPhysicalGroup(3, vols, name="domain")
    gmsh.model.addPhysicalGroup(3, keep, name="keep")
    gmsh.model.addPhysicalGroup(3, void, name="void")
    gmsh.model.addPhysicalGroup(2, sorted(set(bolt_surfs)), name="bolt")
    gmsh.model.addPhysicalGroup(2, sorted(set(pin_surfs)), name="pin")

    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h/3)
    # small holes need finer elements to stay round
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.model.mesh.generate(3)
    gmsh.write(fname)
    n_tets = sum(len(gmsh.model.mesh.getElements(3, t)[1][0]) for t in vols)
    gmsh.finalize()
    print(f"wrote {fname}: {n_tets} tets, {len(keep)} keep regions, "
          f"{len(void)} void columns, "
          f"{len(set(bolt_surfs))} bolt faces, {len(set(pin_surfs))} pin faces")


if __name__ == "__main__":
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.12
    make_bracket(h=h)