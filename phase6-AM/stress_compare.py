"""von Mises stress comparison: unconstrained vs AM-constrained MBB.

Both designs are thresholded to solid/void and re-solved, then coloured on a
shared scale. NOTE: thresholding is not volume-neutral -- the AM design has
more intermediate density, so more of it snaps to solid. The post-threshold
compliances are therefore NOT a fair comparison; the headline +9.4% figure
comes from the volume-matched SIMP fields.
"""
import numpy as np
import matplotlib.pyplot as plt
from topopt_fast import (element_stiffness_Q4, build_edof, fea_solve,
                         compliance_and_sensitivity, mbb_boundary_conditions)

E_real, nu = 200000.0, 0.3      # steel, MPa
P_real, thick = 10000.0, 10.0   # 10 kN, 10 mm
Lx = 900.0                      # half-span, mm
nelx, nely = 120, 40
h = Lx/nelx


def vm_field(rho, tag):
    rho_t = np.where(rho > 0.5, 1.0, 0.001)
    KE = element_stiffness_Q4(E=1.0, nu=nu)
    edof = build_edof(nelx, nely)
    F, fixed = mbb_boundary_conditions(nelx, nely)
    u = fea_solve(nelx, nely, rho_t, 3.0, KE, edof, fixed, F)
    c, _ = compliance_and_sensitivity(nelx, nely, rho_t, 3.0, KE, edof, u)

    u_real = u*(P_real/(E_real*thick))
    dNdx = np.array([-1, 1, 1, -1])/(2*h)
    dNdy = np.array([-1, -1, 1, 1])/(2*h)
    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2*i] = dNdx[i]; B[1, 2*i+1] = dNdy[i]
        B[2, 2*i] = dNdy[i]; B[2, 2*i+1] = dNdx[i]
    D = E_real/(1-nu**2)*np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    sig = (D @ B @ (u_real[edof]*h).T).T
    vm = np.sqrt(sig[:,0]**2 - sig[:,0]*sig[:,1] + sig[:,1]**2 + 3*sig[:,2]**2)
    vm = vm.reshape((nely, nelx), order='F')
    vm = np.where(rho_t > 0.5, vm, np.nan)

    print(f"{tag}: thresholded vol={np.mean(rho_t>0.5):.4f} "
          f"c={c:.1f}  peak={np.nanmax(vm):.0f} MPa  "
          f"median={np.nanmedian(vm):.0f} MPa")
    return vm


def main():
    r0 = np.load("r0.npy"); r1 = np.load("r1.npy")
    v0 = vm_field(r0, "unconstrained")
    v1 = vm_field(r1, "AM-constrained")

    vmax = np.nanpercentile(np.concatenate([v0.ravel(), v1.ravel()]), 98)
    cmap = plt.get_cmap("jet").copy(); cmap.set_bad(color="white")

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    for ax, v, t in zip(axes, [v0, v1],
                        ["Unconstrained SIMP",
                         "AM overhang-constrained (45°)"]):
        full = np.concatenate([np.fliplr(v), v], axis=1)
        im = ax.imshow(full, cmap=cmap, vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"{t} — von Mises under {P_real/1000:.0f} kN (steel)")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02, label="von Mises [MPa]")
    fig.savefig("am_stress.png", dpi=150, bbox_inches="tight")
    print(f"shared colour max: {vmax:.0f} MPa (98th percentile)")
    print("saved am_stress.png")


if __name__ == "__main__":
    main()