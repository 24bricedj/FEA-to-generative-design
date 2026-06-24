"""
Analytical reference for the fully-constrained uniform-heating benchmark.

A block heated by dT with all boundaries fully fixed: total strain = 0 but
thermal strain = alpha*dT, so mechanical strain = -alpha*dT, giving uniform
biaxial compression:

    sigma_x = sigma_y = -E alpha dT / (1 - nu),    tau_xy = 0
"""

def constrained_thermal_stress(E, nu, alpha, dT):
    return -E * alpha * dT / (1.0 - nu)