"""MBB beam topology optimization driver."""
import numpy as np
from topopt import (element_stiffness_Q4, fea_solve, compliance_and_sensitivity,
                    sensitivity_filter, mbb_boundary_conditions, plot_design)


def oc_update(nelx, nely, rho, volfrac, dc, move=0.2):
    """Optimality-criteria update: rho_new = rho * sqrt(B), with move limit
    and [0.001, 1] clamping, and lambda found by binary search so the volume
    constraint is satisfied."""
    l1 = 0.0
    l2 = 1e9
    while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-3:
        lmid = 0.5 * (l1 + l2)
        B = -dc / lmid
        rho_new = np.maximum(0.001,
                     np.maximum(rho - move,
                        np.minimum(1.0,
                           np.minimum(rho + move,
                              rho * np.sqrt(B)))))
        if rho_new.sum() > volfrac * nelx * nely:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new



def main():
    nelx, nely = 60, 20
    volfrac, penal, rmin = 0.5, 3.0, 1.5

    KE = element_stiffness_Q4()
    F, fixed = mbb_boundary_conditions(nelx, nely)
    rho = volfrac*np.ones((nely, nelx))

    for it in range(150):
        u = fea_solve(nelx, nely, rho, penal, KE, fixed, F)
        c, dc = compliance_and_sensitivity(nelx, nely, rho, penal, KE, u)
        dc = sensitivity_filter(nelx, nely, rmin, rho, dc)
        rho_old = rho.copy()
        rho = oc_update(nelx, nely, rho, volfrac, dc)
        change = np.abs(rho - rho_old).max()
        print(f"it {it:3d}  c={c:9.4f}  vol={rho.mean():.3f}  change={change:.4f}")
        if change < 0.01:
            print("converged")
            break

    plot_design(rho, f"MBB beam, {volfrac:.0%} volume, p={penal}",
                "mbb_result.png")
    print("saved mbb_result.png")

    from topopt import plot_full_mbb
    plot_full_mbb(rho, f"MBB beam (full span, mirrored), {volfrac:.0%} volume, p={penal}",
                  "mbb_full.png")
    print("saved mbb_full.png")


if __name__ == "__main__":
    main()