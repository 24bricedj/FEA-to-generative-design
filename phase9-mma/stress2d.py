"""
Stress measure for 2D (plane-stress, CST) topology optimization, and its
gradient by the adjoint method. Everything here is what you derived:

  element stress     s_e     = D B_e u_e                       (Phase 2)
  von Mises          vm_e    = sqrt(s_e^T M s_e)
  relaxed stress     sig_e   = rho_e^q vm_e        q < 1: gray can't hide stress
  p-norm             sig_PN  = (sum_e sig_e^P)^(1/P)          smooth max
  weight             w_e     = (sig_e / sig_PN)^(P-1)          = d sig_PN / d sig_e
  adjoint RHS        d sig_PN/du  <-  sum_e w_e rho_e^q B_e^T D M s_e / vm_e
  adjoint solve      K lam = d sig_PN/du
  gradient           d sig_PN/d rho_e = w_e q rho_e^(q-1) vm_e  -  p rho_e^(p-1) lam_e^T K_e u_e
                                         (direct)                   (through u)
(rho here is the FILTERED density -- the physics' density.)
"""
import numpy as np

# von Mises in plane stress:  vm^2 = sx^2 + sy^2 - sx*sy + 3 txy^2  =  s^T M s
M = np.array([[1.0, -0.5, 0.0],
              [-0.5, 1.0, 0.0],
              [0.0, 0.0, 3.0]])


def plane_stress_D(E, nu):
    return E/(1 - nu**2)*np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu)/2]])


def cst_B(nodes, tris):
    """B (ne, 3, 6) for every triangle at once -- your Phase 2 B matrix."""
    x, y = nodes[tris, 0], nodes[tris, 1]                       # (ne, 3)
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
    A2 = (x[:, 1] - x[:, 0])*(y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0])*(y[:, 1] - y[:, 0])
    B = np.zeros((len(tris), 3, 6))
    B[:, 0, 0::2] = b
    B[:, 1, 1::2] = c
    B[:, 2, 0::2] = c
    B[:, 2, 1::2] = b
    return B/A2[:, None, None]


class StressMeasure:
    def __init__(self, B, D, edof, ndof, q=0.5, P=8.0):
        self.B, self.D, self.edof, self.ndof, self.q, self.P = B, D, edof, ndof, q, P
        self.DB = np.einsum('ij,ejk->eik', D, B)                # (ne, 3, 6)

    def evaluate(self, rho, u):
        """Returns sig_PN, element stresses sig_e and vm_e, d sig_PN/du (ndof),
        and the direct part of d sig_PN/d rho (ne)."""
        q, P = self.q, self.P
        ue = u[self.edof]                                       # (ne, 6)
        s = np.einsum('eik,ek->ei', self.DB, ue)                # (ne, 3) element stress
        
        vm = np.sqrt(np.einsum('ei,ij,ej->e', s, M, s) + 1e-12)
        sig = rho**q*vm                                         # relaxed stress
        pn = np.sum(sig**P)**(1/P)                              # p-norm
        
        w = (sig/pn)**(P - 1)
        

        DMs = np.einsum('eik,ij,ej->ek', self.DB, M, s)
        g = DMs * (w*rho**q/vm)[:, None]


        dpn_du = np.zeros(self.ndof)
        np.add.at(dpn_du, self.edof, g)                         # scatter, like assembling F
        direct = w*q*rho**(q - 1)*vm                            # the rho^q factor's own slope
        return pn, sig, vm, dpn_du, direct


def adjoint_term(rho, penal, KE, edof, u, lam):
    """...docstring..."""
    lame = lam[edof]
    ue = u[edof]
    return -penal*rho**(penal-1)*np.einsum('ei,eij,ej->e', lame, KE, ue)
