"""Plotting utilities for the 2D CST solver."""

import matplotlib.pyplot as plt
import numpy as np


def plot_deformed_mesh(nodes, connectivity, u_full, scale=1.0,
                       savepath=None):
    """Plot the undeformed and deformed mesh overlaid.

    Parameters
    ----------
    nodes : ndarray (n_nodes, 2)
        Original node coordinates.
    connectivity : ndarray (n_elements, 3)
        Triangle node indices.
    u_full : ndarray (2 * n_nodes,)
        Full nodal displacement vector, DOF order [u0, v0, u1, v1, ...].
    scale : float
        Displacement magnification factor. Real deflections are often tiny
        relative to the structure, so we exaggerate them to see the shape.
    savepath : str, optional
        If given, save the figure here.
    """
    # reshape the flat DOF vector into (n_nodes, 2): column 0 = u, column 1 = v
    disp = u_full.reshape(-1, 2)
    deformed = nodes + scale * disp

    fig, ax = plt.subplots(figsize=(10, 4))

    # undeformed mesh (light gray)
    _draw_triangles(ax, nodes, connectivity,
                    edgecolor="0.8", lw=0.5, label="Undeformed")
    # deformed mesh (color)
    _draw_triangles(ax, deformed, connectivity,
                    edgecolor="C0", lw=0.7, label="Deformed")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Cantilever deformation (scale = {scale}x)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def _draw_triangles(ax, pts, connectivity, edgecolor, lw, label):
    """Draw triangle edges for a mesh. Internal helper."""
    first = True
    for tri in connectivity:
        # close the triangle by repeating the first vertex
        idx = [tri[0], tri[1], tri[2], tri[0]]
        xy = pts[idx]
        ax.plot(xy[:, 0], xy[:, 1], color=edgecolor, lw=lw,
                label=label if first else None)
        first = False