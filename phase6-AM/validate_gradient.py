"""Validate the overhang-filter adjoint sweep against finite differences.

Uses a toy objective c = sum(w * rho_hat^3) with random weights, so the
gradient can be checked independently of the FEA. Agreement to ~1e-4 confirms
the backward sweep; the residual is finite-difference truncation error.
"""
import numpy as np
from amfilter import overhang_filter, overhang_backward

np.random.seed(3)
nely, nelx, q = 6, 7, 8.0
rho = np.random.rand(nely, nelx)*0.8 + 0.15
w = np.random.rand(nely, nelx)


def obj(r):
    rh, _, _, _ = overhang_filter(r, q)
    return np.sum(w*rh**3)


rh, S, T, R = overhang_filter(rho, q)
g_an = overhang_backward(rho, rh, S, T, R, 3*w*rh**2, q)

eps = 1e-7
g_fd = np.zeros_like(rho)
c0 = obj(rho)
for a in range(nely):
    for b in range(nelx):
        rp = rho.copy(); rp[a, b] += eps
        g_fd[a, b] = (obj(rp) - c0)/eps

rel = np.abs(g_an - g_fd)/(np.abs(g_fd) + 1e-10)
print(f"max relative error:  {rel.max():.3e}")
print(f"mean relative error: {rel.mean():.3e}")