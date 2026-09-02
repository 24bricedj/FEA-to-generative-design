"""AM overhang filter: printable density and its adjoint sweep."""
import numpy as np

EPS = 1e-4      # smin smoothing; too small makes the kink unresolvable


def overhang_filter(rho, q=8.0):
    nely, nelx = rho.shape
    rho_hat = np.zeros_like(rho)
    S = np.ones_like(rho)      # support factor
    T = np.zeros_like(rho)     # the power-sum inside S
    R = np.ones_like(rho)      # sqrt((rho-S)^2 + EPS) from the smin

    # bottom row sits on the build plate: always fully supported
    rho_hat[nely-1, :] = rho[nely-1, :]

    # work upward: row n is above row n+1
    for n in range(nely-2, -1, -1):
        below = rho_hat[n+1, :]                              # the row physically beneath
        left  = np.concatenate([[0.0], below[:-1]])          # below[i-1], 0 off-edge
        right = np.concatenate([below[1:], [0.0]])           # below[i+1], 0 off-edge

        t = left**q + right**q + below**q          # (A) power-sum of the three supporters
        s = t**(1.0/q)          # (B) the q-th root of t
        s = np.minimum(s, 1.0)              # support can't exceed "fully supported"

        d = rho[n, :] - s          # (C) rho[n,:] minus s
        r = np.sqrt(d*d + EPS)
        m = 0.5 * (rho[n, :] + s - r)          # (D) the smooth min of rho[n,:] and s

        T[n, :] = t
        S[n, :] = s
        R[n, :] = r
        rho_hat[n, :] = np.maximum(m, 0.001)

    return rho_hat, S, T, R



def overhang_backward(rho, rho_hat, S, T, R, dc_dhat, q=8.0):
    nely, nelx = rho.shape
    Lam = np.zeros_like(rho)
    dc_drho = np.zeros_like(rho)

    # top row: nothing above it, so only the direct physics term
    Lam[0, :] = dc_dhat[0, :]

    for n in range(0, nely):
        if n > 0:
            # --- contribution flowing down from the row above (row n-1) ---
            d = rho[n-1, :] - S[n-1, :]
            dsmin_dS = 0.5 * (1 + d/R[n-1, :])    # (E) 0.5*(1 + d/R[n-1,:])

            # dS/d(rho_hat below) needs T^(1/q - 1); guard T=0
            Tp = np.where(T[n-1, :] > 0, T[n-1, :]**(1.0/q - 1.0), 0.0)
            active = (S[n-1, :] < 1.0)            # zero gradient where S was clamped

            W = Lam[n-1, :] * dsmin_dS * Tp * active          # (F) Lam[n-1,:] * dsmin_dS * Tp * active

            # element (n,i) supports (n-1, i-1), (n-1, i), (n-1, i+1):
            # sum W[j] for j in {i-1, i, i+1}
            Wl = np.concatenate([W[1:], [0.0]])   # W[i+1]
            Wr = np.concatenate([[0.0], W[:-1]])  # W[i-1]
            stencil = Wl + W + Wr

            Lam[n, :] = dc_dhat[n, :] + rho_hat[n, :]**(q-1.0) * stencil  # (G) direct term + rho_hat[n,:]**(q-1) * stencil

        # --- convert accumulated Lam into design-variable sensitivity ---
        if n < nely-1:
            d = rho[n, :] - S[n, :]
            dsmin_drho = 0.5 * (1 - d/R[n, :])   # (H) 0.5*(1 - d/R[n,:])
        else:
            dsmin_drho = np.ones(nelx)            # bottom row: rho_hat = rho exactly

        dc_drho[n, :] = Lam[n, :] * dsmin_drho   # (I) Lam[n,:] * dsmin_drho

    return dc_drho