"""
Vectorized 3D tet setup for big meshes (hundreds of thousands of elements).

Same math as your element_stiffness_tet (k = B^T D B V) -- but built for ALL
elements at once with numpy instead of a Python loop, which matters at 645k
tets (a loop takes minutes; this takes seconds).
"""
import os
import sys
import numpy as np

# reuse the Phase 7 modules (fast_solve, fem3d) without copying them
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "phase7-mesh-input"))
from fem3d import iso_D_3d          # noqa: E402  (your 6x6 D)


def precompute_3d_fast(nodes, tets, E, nu):
    """Returns KE (ne,12,12), V (ne,), edof (ne,12) int32, and the tets with
    any inverted elements re-ordered so every volume is positive."""
    tets = tets.astype(np.int64).copy()
    C = np.concatenate([np.ones((len(tets), 4, 1)), nodes[tets]], axis=2)  # [1 x y z]
    V = np.linalg.det(C)/6.0
    flip = V < 0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    C[flip] = C[flip][:, [0, 2, 1, 3]]
    V = np.abs(V)
    G = np.linalg.inv(C)[:, 1:, :]          # (ne, 3, 4): dN_i/dx, dN_i/dy, dN_i/dz

    B = np.zeros((len(tets), 6, 12))
    for i in range(4):                      # your B_i blocks, all elements at once
        bx, by, bz = G[:, 0, i], G[:, 1, i], G[:, 2, i]
        B[:, 0, 3*i] = bx
        B[:, 1, 3*i + 1] = by
        B[:, 2, 3*i + 2] = bz
        B[:, 3, 3*i], B[:, 3, 3*i + 1] = by, bx
        B[:, 4, 3*i + 1], B[:, 4, 3*i + 2] = bz, by
        B[:, 5, 3*i], B[:, 5, 3*i + 2] = bz, bx
    D = iso_D_3d(E, nu)
    KE = np.einsum('eki,kl,elj->eij', B, D, B)*V[:, None, None]
    edof = (np.repeat(3*tets, 3, axis=1) + np.tile([0, 1, 2], 4)).astype(np.int32)
    return KE, V, edof, tets