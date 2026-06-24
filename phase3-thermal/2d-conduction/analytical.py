"""
Analytical reference for the 2D sinusoidal-edge conduction benchmark.

Laplace's equation on [0,W] x [0,H] with:
    bottom (y=0): T = sin(pi x / W)
    top, left, right: T = 0

Separation of variables gives the exact solution:

    T(x,y) = sin(pi x / W) * sinh(pi (H - y)/W) / sinh(pi H / W)
"""
import numpy as np

def T_exact(x, y, W, H):
    return np.sin(np.pi*x/W) * np.sinh(np.pi*(H-y)/W) / np.sinh(np.pi*H/W)