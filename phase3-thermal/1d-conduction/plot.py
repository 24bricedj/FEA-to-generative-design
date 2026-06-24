"""Plotting utilities for the 1D thermal solver."""

import matplotlib.pyplot as plt
import numpy as np


def plot_temperature_profile(nodes, T_fea, T_exact_fn, L, title, savepath=None):
    """Plot FEA nodal temperatures vs the analytical profile.

    Parameters
    ----------
    nodes : ndarray
        Node positions.
    T_fea : ndarray
        FEA nodal temperatures.
    T_exact_fn : callable
        Function x -> T_exact(x).
    L : float
        Bar length.
    title : str
        Plot title.
    savepath : str, optional
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x_fine = np.linspace(0, L, 200)
    ax.plot(x_fine, T_exact_fn(x_fine), "k-", lw=2, label="Analytical")
    ax.plot(nodes, T_fea, "o-", color="C3", lw=1.5, ms=8,
            label=f"FEA ({len(nodes) - 1} elements)")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax