"""
MMA for:   minimize f0(rho)   subject to   f1(rho) <= 0

The same method as mma.py (your asymptotes are reused unchanged), generalized
to gradients of EITHER sign, which stress needs:
    positive slope  ->  p/(U - rho)          negative slope  ->  q/(rho - L)
    p = (U - rho)^2 max(df, 0)               q = (rho - L)^2 max(-df, 0)

Each element's update keeps your closed form,
    s = sqrt(P/Q),   rho = (U + s L)/(1 + s),
    P = p0 + lam p1  (everything pushing rho down)
    Q = q0 + lam q1  (everything pushing rho up),
and lam is found by bisection -- checking the MMA APPROXIMATION of f1, never
a new FEA solve.
"""
import numpy as np
from mma import MMA


class MMAConstrained(MMA):
    def update_c(self, x, df0, f1, df1, keep=None):
        """x: design; df0: objective gradient; f1, df1: constraint value and
        gradient (feasible when f1 <= 0). Returns the new design."""
        self.asymptotes(x)
        L, U = self.L, self.U
        alpha = np.maximum.reduce([np.full(self.n, self.xmin), L + 0.1*(x - L), x - self.move])
        beta = np.minimum.reduce([np.full(self.n, self.xmax), U - 0.1*(U - x), x + self.move])

        # objective terms, with Svanberg's small regularization (keeps every
        # element's subproblem strictly convex even where a slope is ~0)
        reg = 1e-5/(self.xmax - self.xmin)
        p0 = (U - x)**2*(1.001*np.maximum(df0, 0) + 0.001*np.maximum(-df0, 0) + reg)
        q0 = (x - L)**2*(0.001*np.maximum(df0, 0) + 1.001*np.maximum(-df0, 0) + reg)

        p1 = (U - x)**2 * np.maximum(df1, 0)
        q1 = (x - L)**2 * np.maximum(-df1, 0)

        # the MMA approximation of f1, built to equal f1 at the current x
        r1 = f1 - np.sum(p1/(U - x) + q1/(x - L))

        def f1_approx(xn):
            return r1 + np.sum(p1/(U - xn) + q1/(xn - L))

        def x_of(lam):
            s = np.sqrt((p0 + lam*p1)/(q0 + lam*q1))
            xn = np.clip((U + s*L)/(1 + s), alpha, beta)
            if keep is not None:
                xn[keep] = 1.0
            return xn

        if f1_approx(x_of(0.0)) <= 0:
            x_new = x_of(0.0)
        else:
            l1, l2 = 0.0, 1e9
            for _ in range(100):
                lmid = 0.5*(l1 + l2)
                if f1_approx(x_of(lmid)) > 0:
                    l1 = lmid
                else:
                    l2 = lmid
            x_new = x_of(l2)

        self.x2 = self.x1
        self.x1 = x.copy()
        return x_new