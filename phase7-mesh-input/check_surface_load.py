"""Check surface_load against hand calculations."""
import numpy as np
from fem3d_solve import surface_load

# --- 1. your hand calc: triangles abc (area 3) and bcd (area 1), F = -8 in y
#   a=(0,0,0)  b=(3,0,0)  c=(0,2,0)  d=(1.5,1.5,0)... built so areas are 3 and 1
a, b, c = [0, 0, 0], [3, 0, 0], [0, 2, 0]
# put d on the far side of edge bc so triangle bcd has area exactly 1
mid = np.array([1.5, 1.0, 0]); n = np.array([2, 3, 0])/np.sqrt(13)
d = mid + n*(2*1/np.sqrt(13))            # height h = 2A/|bc|, |bc| = sqrt(13)
nodes = np.array([a, b, c, d], dtype=float)
faces = np.array([[0, 1, 2], [1, 2, 3]])
F = surface_load(nodes, faces, 3*len(nodes), direction=1, total=-8.0)
Fy = F[1::3]
print("hand-calc case (y-forces on a, b, c, d):", np.round(Fy, 4))
print("   sum =", round(Fy.sum(), 6), "  (should be -8)")

# --- 2. a unit square split into many uneven triangles: total must survive,
#        and the force per unit area must be the same everywhere
rng = np.random.default_rng(0)
from scipy.spatial import Delaunay
pts = np.vstack([[0, 0], [1, 0], [0, 1], [1, 1], rng.random((40, 2))])
tri = Delaunay(pts).simplices
nodes = np.column_stack([pts, np.zeros(len(pts))])
F = surface_load(nodes, tri, 3*len(nodes), direction=2, total=5.0)
print("random square mesh: total z-force =", round(F[2::3].sum(), 6),
      " (should be 5)   x,y untouched:", np.allclose(F[0::3], 0) and np.allclose(F[1::3], 0))