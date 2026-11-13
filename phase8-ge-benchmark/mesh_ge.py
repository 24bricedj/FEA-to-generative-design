"""
Mesh GE's design envelope for topology optimization.

    python mesh_ge.py [h]          # h = target element size in mm (default 2.5)
    -> ge_envelope.msh

Geometry: Dolfiny's STEP of the GE challenge envelope (pin along x, vertical +z,
"horizontal out" = -y), download from
  https://github.com/fenics-dolfiny/dolfiny/tree/main/demo/structural_optimisation
into data/ge_bracket_rotated_nonds.step.

The envelope IS the original GE bracket. Its shape already leaves out the
clevis gap and the bolt-head clearance pockets, so no "void" regions are needed:
space that is not meshed can never get material.

Physical groups:
  "domain" (3D) every tet
  "keep"   (3D) the four bolt bosses + a ring of material around each pin hole
  "bolt"   (2D) the bolt faces (hole walls, washer face, seat)   -> supports
  "pin"    (2D) the two pin-hole walls (both lugs)                -> loads
"""
import sys
import numpy as np
import gmsh

STEP = "data/ge_bracket_rotated_nonds.step"
PIN_Y, PIN_Z, PIN_R = -56.85, 44.62, 9.55    # pin axis (along x) and hole radius
LUGS = [(-63.6, -55.9), (-92.1, -84.5)]      # x-extent of the two clevis lugs


def main(h=2.5, ring=4.0, fname="ge_envelope.msh"):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(STEP)
    occ = gmsh.model.occ
    occ.synchronize()

    # remember the colored faces by their centers (tags change after fragment)
    def com(d, t):
        return np.array(occ.getCenterOfMass(d, t))
    red, pinc = [], []
    for d, t in gmsh.model.getEntities(2):
        c = tuple(gmsh.model.getColor(d, t))[:3]
        if c == (255, 0, 0):
            red.append(com(d, t))
        elif c in [(0, 255, 0), (0, 0, 255)]:
            pinc.append(com(d, t))

    vols = [t for d, t in gmsh.model.getEntities(3)]
    body = max(vols, key=lambda t: occ.getMass(3, t))
    bosses = [t for t in vols if t != body]

    # keep rings around the pin holes, clipped to the body
    rings = []
    for x0, x1 in LUGS:
        c = occ.addCylinder(x0, PIN_Y, PIN_Z, x1 - x0, 0, 0, PIN_R + ring)
        r, _ = occ.intersect([(3, c)], [(3, body)], removeObject=True, removeTool=False)
        rings += [t for d, t in r]

    # one conforming mesh: split everything where the pieces touch
    out, pieces = occ.fragment([(3, t) for t in [body] + bosses + rings], [])
    occ.synchronize()
    allv = [t for d, t in gmsh.model.getEntities(3)]
    keepv = sorted({t for p in pieces[1:] for d, t in p})   # bosses + rings
    designv = [t for t in allv if t not in keepv]

    def match(targets):
        found = []
        for d, t in gmsh.model.getEntities(2):
            c = com(d, t)
            if any(np.linalg.norm(c - q) < 1e-3 for q in targets):
                found.append(t)
        return found
    bolt_s, pin_s = match(red), match(pinc)

    gmsh.model.addPhysicalGroup(3, allv, name="domain")
    gmsh.model.addPhysicalGroup(3, keepv, name="keep")
    gmsh.model.addPhysicalGroup(2, bolt_s, name="bolt")
    gmsh.model.addPhysicalGroup(2, pin_s, name="pin")

    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h/3)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.model.mesh.generate(3)
    gmsh.write(fname)
    n_tets = sum(len(gmsh.model.mesh.getElements(3, t)[1][0]) for t in allv)
    vol = sum(occ.getMass(3, t) for t in allv)
    vkeep = sum(occ.getMass(3, t) for t in keepv)
    gmsh.finalize()
    print(f"wrote {fname}: {n_tets} tets | envelope {vol:.0f} mm^3, keep {vkeep:.0f} mm^3 "
          f"({vkeep/vol:.1%}) | {len(designv)} design + {len(keepv)} keep volumes | "
          f"{len(bolt_s)} bolt faces, {len(pin_s)} pin faces")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 2.5)