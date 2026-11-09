"""
Fast 3D solver: conjugate gradient (CG) preconditioned with algebraic
multigrid (AMG), with a warm start from the previous iteration.

Why: a direct solve (spsolve) factorizes K, and in 3D the fill-in makes the
cost grow like N^2.3 -- 4 s at 24k DOFs, ~5 min at 144k. CG only multiplies
K by vectors (cost ~N per step), and AMG keeps the number of steps small
(30-50) even though void elements are a billion times softer than solid ones.

Needs:  pip install pyamg      (or: conda install -c conda-forge pyamg)
"""
import numpy as np
import pyamg
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg


def rigid_body_modes(nodes):
    B = np.zeros((3*len(nodes), 6))
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]

    # translations: every node moves 1 in one direction
    B[0::3, 0] = 1        # col 0: x-DOFs = 1
    B[1::3, 1] = 1        # col 1: y-DOFs = 1
    B[2::3, 2] = 1        # col 2: z-DOFs = 1

    # col 3: rotation about z  ->  u = (-y, x, 0)
    B[0::3, 3] = -y
    B[1::3, 3] = x
                          # z-motion is 0, already zero

    # col 4: rotation about x  ->  u = ( ___ , ___ , ___ )
    B[1::3, 4] = -z    # y-motion
    B[2::3, 4] = y      # z-motion

    # col 5: rotation about y  ->  u = ( ___ , ___ , ___ )
    B[0::3, 5] = z      # x-motion
    B[2::3, 5] = -x      # z-motion

    return B

def _cg(A, b, x0, M, tol, maxiter):
    """scipy renamed CG's tolerance argument (tol -> rtol); support both."""
    it = [0]
    def count(_):
        it[0] += 1
    try:
        x, info = cg(A, b, x0=x0, rtol=tol, M=M, maxiter=maxiter, callback=count)
    except TypeError:
        x, info = cg(A, b, x0=x0, tol=tol, M=M, maxiter=maxiter, callback=count)
    return x, info, it[0]


class FastSolver:
    """Drop-in replacement for fea_solve_3d / fea_solve_3d_multi.

    Build it once (before the optimization loop), then call .solve(rho, penal, F)
    every iteration. F may be one load vector (ndof,) or several (ndof, n_cases).
    """

    def __init__(self, KE, edof, ndof, fixed, nodes, tol=1e-8, maxiter=1000):
        self.KE, self.ndof, self.tol, self.maxiter = KE, ndof, tol, maxiter
        n = edof.shape[1]
        # the sparsity pattern never changes -- build the index arrays once
        self.iK = np.repeat(edof, n, axis=1).ravel()
        self.jK = np.tile(edof, (1, n)).ravel()
        self.free = np.setdiff1d(np.arange(ndof), fixed)
        self.B = rigid_body_modes(nodes)[self.free]
        self.U_prev = None              # warm start
        self.iters = []                 # CG steps per load case, last solve

    def solve(self, rho, penal, F):
        single = (F.ndim == 1)
        Fs = F[:, None] if single else F
        sK = (self.KE*(rho**penal)[:, None, None]).ravel()
        K = coo_matrix((sK, (self.iK, self.jK)),
                       shape=(self.ndof, self.ndof)).tocsr()
        Kf = K[self.free][:, self.free]
        # the AMG hierarchy depends only on K: build it ONCE per iteration and
        # reuse it for every load case (same idea as factorizing once)
        ml = pyamg.smoothed_aggregation_solver(Kf, B=self.B, max_coarse=500,
                                               coarse_solver="splu")
        M = ml.aspreconditioner(cycle="V")

        U = np.zeros_like(Fs)
        self.iters = []
        for i in range(Fs.shape[1]):
            x0 = None if self.U_prev is None else self.U_prev[self.free, i]
            x, info, nit = _cg(Kf, Fs[self.free, i], x0, M, self.tol, self.maxiter)
            if info != 0:
                print(f"  warning: CG did not converge for case {i} ({nit} steps)")
            U[self.free, i] = x
            self.iters.append(nit)
        self.U_prev = U.copy()
        return U[:, 0] if single else U