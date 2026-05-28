"""Plotting utilities for the FEA solver."""

import matplotlib.pyplot as plt
import numpy as np


def plot_displacement_comparison(nodes, u_fea, u_exact_fn, L, savepath=None):
    """Plot FEA nodal displacements alongside the analytical solution.

    Parameters
    ----------
    nodes : ndarray
        Node positions.
    u_fea : ndarray
        FEA nodal displacements (same length as nodes).
    u_exact_fn : callable
        Function x -> u_exact(x).
    L : float
        Bar length.
    savepath : str, optional
        If given, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Analytical solution on a fine grid
    x_fine = np.linspace(0, L, 200)
    ax.plot(x_fine, u_exact_fn(x_fine), "k-", lw=2, label="Analytical")

    # FEA solution: piecewise linear between nodes
    ax.plot(nodes, u_fea, "o-", color="C0", lw=1.5, ms=8,
            label=f"FEA ({len(nodes) - 1} elements)")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("u(x) [m]")
    ax.set_title("1D bar displacement: FEA vs analytical")
    ax.legend()
    ax.grid(alpha=0.3)

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax