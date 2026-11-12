"""Check torsion_load: it must be a pure couple.

    python check_torsion.py bracket.msh
"""
import sys
import numpy as np
from mesh_io import read_mesh_3d
from optimize_bracket import torsion_load, YP

fname = sys.argv[1] if len(sys.argv) > 1 else "bracket.msh"
nodes, tets, sets, faces = read_mesh_3d(fname)
ndof = 3*len(nodes)
M = 1.0
F = torsion_load(nodes, faces["pin"], ndof, M, np.array([2.0, YP, 0.0]))

f = F.reshape(-1, 3)                          # one (fx, fy, fz) row per node
center = np.array([2.0, YP, 1.0])             # middle of the pin
mom = np.cross(nodes - center, f).sum(axis=0) # sum of r x f
total_push = np.abs(f[:, 1]).sum()/2          # size of each half of the couple

print(f"net force      (Fx, Fy, Fz) = ({f[:,0].sum():+.4f}, {f[:,1].sum():+.4f}, {f[:,2].sum():+.4f})")
print(f"   -> net Fy is {abs(f[:,1].sum())/total_push:.1%} of each half's push "
      f"(should be ~1% or less: only mesh asymmetry)")
print(f"net moment     (Mx, My, Mz) = ({mom[0]:+.4f}, {mom[1]:+.4f}, {mom[2]:+.4f})")
print(f"   -> Mx should be {M:+.4f}; My, Mz should be ~0")
print(f"check vs. your formula: each half pushes about 1.5*M/W = {1.5*M/2:.3f}; "
      f"code gives {total_push:.3f}")