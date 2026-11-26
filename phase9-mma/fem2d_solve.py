"""
Plumbing: one stiffness assembly + one factorization, then as many solves as
you like. The stress gradient needs two solves per iteration with the SAME K:
    K u   = F              (the structure)
    K lam = d sig_PN / du  (the adjoint)
so factor once (splu) and reuse it -- the same trick as Phase 7's multi-load.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


class Solver2D:
    def __init__(self, KE, edof, ndof, fixed, Emin=1e-9):
        self.KE, self.edof, self.ndof, self.Emin = KE, edof, ndof, Emin
        self.free = np.setdiff1d(np.arange(ndof), fixed)
        self.iK = np.repeat(edof, 6, axis=1).ravel()
        self.jK = np.tile(edof, (1, 6)).ravel()

    def factor(self, rho, penal):
        E = self.Emin + (1 - self.Emin)*rho**penal
        sK = (self.KE*E[:, None, None]).ravel()
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
        self.lu = splu(K[self.free][:, self.free])

    def solve(self, F):
        u = np.zeros(self.ndof)
        u[self.free] = self.lu.solve(F[self.free])
        return u