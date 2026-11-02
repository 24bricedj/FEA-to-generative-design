"""Validate the tet element against its physical properties."""
import numpy as np
from fem3d import element_stiffness_tet

X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])

# a second tet: rotated, scaled, and shifted -- the zero modes must survive
R = np.array([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]])
X2 = X @ R.T*2.5 + np.array([3.0, -1.0, 7.0])

for name, coords in [("unit tet", X), ("rotated/scaled tet", X2)]:
    k = element_stiffness_tet(coords, E=1.0, nu=0.3)
    ev = np.linalg.eigvalsh(k)
    tol = 1e-9*ev.max()
    print(f"{name}:")
    print(f"  shape:       {k.shape}")
    print(f"  symmetric:   {np.allclose(k, k.T)}")
    print(f"  rank:        {np.linalg.matrix_rank(k, tol=tol)}")
    print(f"  zero eigs:   {np.sum(np.abs(ev) < tol)}")
    print(f"  min eig:     {ev.min():.2e}  (should be ~0, never meaningfully negative)")

# a rigid rotation about z stores no energy:  u = (-y, x, 0) at each node
k = element_stiffness_tet(X, E=1.0, nu=0.3)
u = np.column_stack([-X[:, 1], X[:, 0], np.zeros(4)]).ravel()
print(f"\nenergy of a rigid z-rotation: {u @ k @ u:.2e}")