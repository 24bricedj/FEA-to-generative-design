"""Check FastSolver against the direct solver on the bracket.

    python mesh_bracket3d.py 0.12
    python check_fast_solve.py bracket.msh
"""
import sys
import time
import numpy as np
from mesh_io import read_mesh_3d, read_element_sets
from fem3d_solve import precompute_3d, fea_solve_3d
from fast_solve import rigid_body_modes, FastSolver
from optimize_bracket import pin_load, YP

fname = sys.argv[1] if len(sys.argv) > 1 else "bracket.msh"
nodes, tets, sets, faces = read_mesh_3d(fname)
es = read_element_sets(fname)
KE, V, edof, cent = precompute_3d(nodes, tets)
ndof = 3*len(nodes)

# --- 1. rigid-body modes: K times each mode must be ~0 (no force needed)
from scipy.sparse import coo_matrix
sK = KE.ravel()
K = coo_matrix((sK, (np.repeat(edof, 12, axis=1).ravel(),
                     np.tile(edof, (1, 12)).ravel())), shape=(ndof, ndof)).tocsr()
B = rigid_body_modes(nodes)
print("rigid-body modes  |K b| / (|K| |b|)  -- should all be ~1e-15:")
scale = abs(K).max()
for k in range(6):
    print(f"  column {k}: {np.linalg.norm(K @ B[:, k])/(scale*np.linalg.norm(B[:, k])):.1e}")

# --- 2. same answer as the direct solver, and faster
b = sets["bolt"]
fixed = np.concatenate([3*b, 3*b + 1, 3*b + 2])
F = pin_load(nodes, faces["pin"], ndof, (0, 1, 0), 1.0, np.array([2.0, YP, 0.0]))
rho = np.full(len(tets), 0.25)
rho[es["keep"]] = 1.0
rho[es["void"]] = 0.001

t = time.time(); u_direct = fea_solve_3d(rho, 3.0, KE, edof, ndof, fixed, F)
t_direct = time.time() - t
solver = FastSolver(KE, edof, ndof, fixed, nodes)
t = time.time(); u_fast = solver.solve(rho, 3.0, F)
t_fast = time.time() - t
err = np.linalg.norm(u_fast - u_direct)/np.linalg.norm(u_direct)
print(f"\ndirect: {t_direct:.2f}s   AMG-CG: {t_fast:.2f}s ({solver.iters[0]} CG steps)")
print(f"relative difference: {err:.1e}   (should be < 1e-6)")
print(f"compliance: direct {F @ u_direct:.6f}   fast {F @ u_fast:.6f}")