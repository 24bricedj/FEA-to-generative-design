"""
Method of Moving Asymptotes (Svanberg, 1987) -- one volume constraint.

At every iteration each function is replaced by a reciprocal approximation
that matches its value and slope at the current design:

    objective (slope < 0):   q_j / (rho_j - L_j)        q_j = -(rho_j - L_j)^2 dc_j
    volume    (slope > 0):   p_j / (U_j - rho_j)        p_j =  (U_j - rho_j)^2 V_j

The walls L_j < rho_j < U_j bracket each variable. The approximation is
separable, so with a multiplier lam on the volume constraint every element
solves its own tiny problem in closed form:

    s_j   = sqrt(lam * p_j / q_j)
    rho_j = (U_j + s_j L_j) / (1 + s_j)          then clamp to [alpha_j, beta_j]

and lam is found by bisection on the volume (volume falls monotonically as
lam rises). The walls move every iteration: in (x0.7) for elements that
oscillate, out (x1.2) for elements moving steadily.
"""
import numpy as np


class MMA:
    def __init__(self, n, xmin=0.001, xmax=1.0, move=0.2,
                 asyinit=0.5, asydecr=0.7, asyincr=1.2):
        self.n, self.xmin, self.xmax, self.move = n, xmin, xmax, move
        self.asyinit, self.asydecr, self.asyincr = asyinit, asydecr, asyincr
        self.x1 = None          # design one iteration ago
        self.x2 = None          # design two iterations ago
        self.L = None
        self.U = None

    def asymptotes(self, x):
        R = self.xmax - self.xmin
        if self.x2 is None:
            self.L = x - self.asyinit*R
            self.U = x + self.asyinit*R
        else:
            sgn = (x - self.x1)*(self.x1 - self.x2)      # + steady, - oscillating
            gamma = np.where(sgn < 0, self.asydecr, np.where(sgn > 0, self.asyincr, 1.0))
            self.L = x - gamma*(self.x1 - self.L)
            self.U = x +gamma*(self.U - self.x1)

        # keep the walls neither glued to x nor absurdly far (standard safeguard)
        R = self.xmax - self.xmin
        self.L = np.clip(self.L, x - 10*R, x - 0.01*R)
        self.U = np.clip(self.U, x + 0.01*R, x + 10*R)

    def update(self, x, dc, V, v_target, keep=None, void=None):
        """One MMA step for: minimize c  s.t.  sum(rho V) <= v_target.

        x  : current densities          dc : d(compliance)/d(rho), all <= 0
        V  : element volumes            v_target : allowed material volume
        """
        self.asymptotes(x)
        L, U = self.L, self.U
        # the box each element may move in: inside the walls, inside [xmin,
        # xmax], and inside a plain move limit (belt and braces)
        alpha = np.maximum.reduce([np.full(self.n, self.xmin), L + 0.1*(x - L), x - self.move])
        beta = np.minimum.reduce([np.full(self.n, self.xmax), U - 0.1*(U - x), x + self.move])

        q = -(x - L)**2*dc + 1e-30
        p = (U - x)**2*V
        l1, l2 = 0.0, 1e9
        while (l2 - l1)/(l1 + l2 + 1e-30) > 1e-4:
            lmid = 0.5*(l1 + l2)
            s = np.sqrt(lmid*p/q)
            x_new = np.clip((U + s*L)/(1 + s), alpha, beta)
            if keep is not None:
                x_new[keep] = 1.0
            if void is not None:
                x_new[void] = self.xmin
            if (x_new*V).sum() > v_target:
                l1 = lmid
            else:
                l2 = lmid

        # history for the asymptote rule
        self.x2 = self.x1
        self.x1 = x.copy()
        return x_new