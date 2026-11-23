"""Check the density filter on random data.

1. a uniform field stays uniform (every row of W averages to 1)
2. the "dot-product test": for ANY vectors x and g,
       (W x) . g  ==  x . (W^T g)
   This is the definition of the transpose. If backward() is wrong, these
   two numbers disagree -- it's the standard way to test any backward pass.
"""
import numpy as np
from density_filter import DensityFilter

rng = np.random.default_rng(0)
cent = rng.random((300, 2))           # 300 random "elements" in a unit square
V = rng.random(300) + 0.5             # uneven element sizes
df = DensityFilter(cent, V, rmin=0.15)

print("uniform 0.5 in -> max error out:", np.abs(df.forward(np.full(300, 0.5)) - 0.5).max())
x, g = rng.random(300), rng.random(300)
a, b = df.forward(x) @ g, x @ df.backward(g)
print(f"dot-product test:  (Wx).g = {a:.10f}   x.(W^T g) = {b:.10f}   "
      f"relative difference {abs(a - b)/abs(a):.1e}  (should be ~1e-16)")