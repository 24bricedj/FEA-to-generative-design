"""MBB beam at 180x60, with von Mises stress visualization."""
import numpy as np
import matplotlib.pyplot as plt
from topopt_fast import (element_stiffness_Q4, build_edof, build_filter,
                         fea_solve, compliance_and_sensitivity, apply_filter,
                         mbb_boundary_conditions)
from optimize import oc_update      # YOUR update rule


def von_mises_field(nelx, nely, rho_t, u_unit, edof, E_real, nu, P_real, thick, h):
    """Element von Mises stress in the thresholded (solid/void) structure."""
    scale = P_real/(E_real*thick)
    u_real = u_unit*scale
    dNdx = np.array([-1, 1, 1, -1])/(2*h)
    dNdy = np.array([-1, -1, 1, 1])/(2*h)
    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2*i] = dNdx[i]; B[1, 2*i+1] = dNdy[i]
        B[2, 2*i] = dNdy[i]; B[2, 2*i+1] = dNdx[i]
    D = E_real/(1-nu**2)*np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    ue = u_real[edof]*h
    sig = (D @ B @ ue.T).T
    vm = np.sqrt(sig[:,0]**2 - sig[:,0]*sig[:,1] + sig[:,1]**2 + 3*sig[:,2]**2)
    vm = vm.reshape((nely, nelx), order='F')
    return np.where(rho_t > 0.5, vm, np.nan)


def main():
    nelx, nely = 180, 60
    volfrac, penal = 0.5, 3.0
    rmin = 4.5              # 1.5 x 3, scaled with the mesh (mesh-independence)

    # real units for the stress plot
    E_real, nu = 200000.0, 0.3   # steel, MPa
    P_real, thick = 10000.0, 10.0  # 10 kN, 10 mm thick
    Lx = 900.0                    # half-span, mm

    KE = element_stiffness_Q4()
    edof = build_edof(nelx, nely)
    F, fixed = mbb_boundary_conditions(nelx, nely)
    H, Hs = build_filter(nelx, nely, rmin)
    rho = volfrac*np.ones((nely, nelx))

    for it in range(200):
        u = fea_solve(nelx, nely, rho, penal, KE, edof, fixed, F)
        c, dc = compliance_and_sensitivity(nelx, nely, rho, penal, KE, edof, u)
        dc = apply_filter(H, Hs, rho, dc)
        rho_old = rho.copy()
        rho = oc_update(nelx, nely, rho, volfrac, dc)
        change = np.abs(rho-rho_old).max()
        if it % 10 == 0:
            print(f"it {it:3d}  c={c:9.3f}  vol={rho.mean():.3f}  change={change:.4f}")
        if change < 0.01:
            print(f"converged at iteration {it}, c={c:.3f}")
            break

    # --- design plot (full span, mirrored), density colormap ---
    full = np.concatenate([np.fliplr(rho), rho], axis=1)
    fig, ax = plt.subplots(figsize=(14, 2.8))
    im = ax.imshow(full, cmap="jet", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"MBB beam, {nelx}x{nely}, {volfrac:.0%} volume, p={penal}")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("density ρ")
    fig.savefig("mbb_full_180.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # --- threshold, re-solve, stress ---
    rho_t = np.where(rho > 0.5, 1.0, 0.001)
    u_t = fea_solve(nelx, nely, rho_t, penal, KE, edof, fixed, F)
    c_t, _ = compliance_and_sensitivity(nelx, nely, rho_t, penal, KE, edof, u_t)
    print(f"compliance: SIMP field {c:.3f} -> thresholded {c_t:.3f} "
          f"({100*(c_t-c)/c:+.1f}%)")

    vm = von_mises_field(nelx, nely, rho_t, u_t, edof,
                         E_real, nu, P_real, thick, Lx/nelx)
    print(f"peak von Mises {np.nanmax(vm):.0f} MPa (point-load singularity), "
          f"median {np.nanmedian(vm):.0f} MPa")

    full_vm = np.concatenate([np.fliplr(vm), vm], axis=1)
    vmax = np.nanpercentile(vm, 98)
    cmap = plt.get_cmap("jet").copy(); cmap.set_bad(color="white")
    fig, ax = plt.subplots(figsize=(14, 3.0))
    im = ax.imshow(full_vm, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"MBB beam - von Mises stress under {P_real/1000:.0f} kN load "
                 f"(steel, E={E_real/1000:.0f} GPa)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("von Mises [MPa]")
    fig.savefig("mbb_stress.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("saved mbb_full_180.png, mbb_stress.png")


if __name__ == "__main__":
    main()