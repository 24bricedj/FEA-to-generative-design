"""
Density filter: the physics runs on filtered densities, and the gradient is
chained back EXACTLY (no heuristic), which is what MMA needs.

Forward (each filtered density is a volume-weighted average of its neighbours):

    rho_t_i = sum_j H_ij V_j rho_j / sum_j H_ij V_j        i.e.  rho_t = W rho

with H your Phase 7 neighbour weights (rmin - distance). Every row of W sums
to 1, so a uniform field stays uniform.

Backward (chain rule -- column j of W collects every path from rho_j):

    d(.)/d rho_j = sum_i W_ij d(.)/d rho_t_i                i.e.  W^T g

Applied to dc/d(rho_t) it gives the exact dc/d(rho); applied to the element
volumes V it gives dv, the gradient of the filtered volume (and, because the
filter is linear, filtered volume = sum(dv * rho) exactly).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "phase7-mesh-input"))
from fem_mesh import build_filter                                 # noqa: E402


class DensityFilter:
    def __init__(self, cent, V, rmin):
        self.H, _ = build_filter(cent, rmin)        # symmetric: H_ij = H_ji
        self.V = V
        self.HV = self.H @ V                         # row sums sum_j H_ij V_j

    def forward(self, rho):

        return (self.H @ (self.V*rho))/self.HV
    

    def backward(self, g):

        return self.V*(self.H @ (g/self.HV))