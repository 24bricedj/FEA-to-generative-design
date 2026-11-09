"""Check the PDE filter.

    python check_pde_filter.py bracket.msh      (use the h = 0.12 mesh)
"""
import sys
import time
import numpy as np
from mesh_io import read_mesh_3d
from fem3d_solve import precompute_3d
from fem_mesh import build_filter, apply_filter
from pde_filter import PDEFilter, element_conduction_tet

fname = sys.argv[1] if len(sys.argv) > 1 else "bracket.msh"
nodes, tets, sets, faces = read_mesh_3d(fname)
KE, V, edof, cent = precompute_3d(nodes, tets)

# 1. element: symmetric, and a uniform temperature makes no heat flow
k = element_conduction_tet(nodes[tets[0]])
print(f"element: shape {k.shape}, symmetric {np.allclose(k, k.T)}, "
      f"max |row sum| = {np.abs(k.sum(axis=1)).max():.1e}  (should be ~0)")

# 2. filtering a constant returns the same constant
t = time.time()
P = PDEFilter(nodes, tets, V, rmin=0.25)
print(f"filter setup: {time.time()-t:.1f}s")
print(f"constant 0.7 in -> max error out: "
      f"{np.abs(P.smooth(np.full(len(tets), 0.7)) - 0.7).max():.1e}  (should be ~1e-15)")

# 3. it agrees with your KD-tree filter on random data
H, Hs = build_filter(cent, 0.25)
rng = np.random.default_rng(1)
rho = np.clip(rng.random(len(tets)), 0.01, 1)
dc = -rng.random(len(tets))*V
a = apply_filter(H, Hs, rho, dc, V)
b = P.apply(rho, dc, V)
print(f"correlation with the KD-tree filter: {np.corrcoef(a, b)[0, 1]:.4f}  (should be > 0.99)")