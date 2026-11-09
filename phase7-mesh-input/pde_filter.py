"""
PDE (Helmholtz) filter -- the scalable replacement for the KD-tree filter.

Solve, on the same tet mesh,

        -r^2 lap(f~) + f~ = f            (insulated boundaries)

which is steady heat conduction with conductivity r^2, a heat source f, and a
loss term proportional to temperature (like a fin). The result f~ is a
distance-weighted average of f over a radius of about r.

Discretized with linear tets:
        (r^2 Kd + M) f~_nodes = T f_elements
        Kd : conduction matrix          (your Phase 3 element, in 3D)
        M  : "mass" matrix for the +f~ term
        T  : spreads each element's source to its 4 nodes, V_e/4 each
then each element's filtered value is the average of its 4 nodes.

The matrix (r^2 Kd + M) depends only on the mesh -- not on rho -- so it is
factorized ONCE for the whole optimization. Memory grows like the number of
nodes, no matter how big rmin is.

Lazarov & Sigmund (2011), "Filters in topology optimization based on
Helmholtz-type differential equations"
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from fem3d import tet_gradients


def element_conduction_tet(coords):
    """4x4 conduction matrix of a linear tet, with conductivity 1.

    YOUR JOB -- the 3D version of your Phase 3 element. There, with
    temperature gradients  grad T = B T_e  (B was 2x3), you got
        k_e = B^T (k) B * t * A.
    Here, with conductivity 1 (r^2 multiplies it later):
      - tet_gradients(coords) returns V and G, where G (3x4) holds the
        gradient of each shape function: grad T = G T_e.  G plays the role of B.
      - "t * A" becomes the tet's volume V.
    So return the 4x4 matrix  G^T G V.
    """
    V, G = tet_gradients(coords)
    return G.T @ G * V


def element_mass_tet(V):
    """4x4 matrix of the integrals of N_i N_j over a linear tet:
    V/20 on the off-diagonals and 2V/20 on the diagonal."""
    return V/20.0*(np.ones((4, 4)) + np.eye(4))


class PDEFilter:
    """Drop-in replacement for build_filter / apply_filter."""

    def __init__(self, nodes, tets, V, rmin):
        r = rmin/(2*np.sqrt(3))          # matches the cone filter's radius rmin
        nn, ne = len(nodes), len(tets)
        ke = np.array([element_conduction_tet(nodes[t]) for t in tets])
        me = np.array([element_mass_tet(v) for v in V])
        iK = np.repeat(tets, 4, axis=1).ravel()
        jK = np.tile(tets, (1, 4)).ravel()
        A = coo_matrix(((r**2*ke + me).ravel(), (iK, jK)), shape=(nn, nn)).tocsc()
        self.lu = splu(A)                 # factorize once, for the whole run
        # T: element -> node, each element hands V_e/4 to each of its nodes
        self.T = coo_matrix((np.repeat(V/4, 4), (tets.ravel(), np.repeat(np.arange(ne), 4))),
                            shape=(nn, ne)).tocsr()
        self.tets = tets

    def smooth(self, f):
        """Filter a per-unit-volume element field f -> element field f~."""
        fn = self.lu.solve(self.T @ f)    # temperature at the nodes
        return fn[self.tets].mean(axis=1) # back to elements

    def apply(self, rho, dc, V):
        """Sensitivity filter, same form as apply_filter: smooth the
        per-unit-volume quantity rho*dc/V, then scale back to element size."""
        return V*self.smooth(rho*dc/V)/np.maximum(1e-3, rho)