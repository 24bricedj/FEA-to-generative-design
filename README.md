# FEA -> Generative Design

A from-scratch computational mechanics stack: finite element analysis -> topology optimization -> ML-accelerated generative design. Each phase is derived by hand, implemented in Python, and validated against analytical solutions before any complexity is added.

---

## Motivation

My motivation for this project comes from a deep love for motorsports and the engineering behind arguably the most complex machines in the world: F1 cars and hypercars. I've been an F1 fan since 2016, after attending my first race at age 10, and ever since I've been infatuated with cars and motorsports. The more I learned about the complexity of high-performance vehicle engineering, the more certain I became that I wanted to build a career in those fields.

Over the last few years I've grown increasingly interested in AI's role in the engineering and design of complex automotive systems. Czinger and their 21C hypercar -- built using additive manufacturing combined with AI-driven generative design -- especially captured my interest. I wanted to understand how they built what they built: they went from nothing to producing one of the fastest hypercars in the world in roughly six years. That manufacturing capability extends well beyond cars. Divergent, the parent company, and their DAPS platform are a major inspiration for this project -- I want to understand how they design the optimized parts now being adopted by top automotive brands.

Nuclear engineering is the other half of my motivation. My dad has spent his career in the nuclear sector, so I grew up immersed in that world, and it's become a real passion of mine -- the audacity of the concept and the elegance of the designs. Many of the same simulation, optimization, and ML strategies I'll explore here (FEA, topology optimization, surrogate modeling) are directly relevant to that field, which is part of why this project is built to serve both.

---

## Project Philosophy

This project is built on three principles:

1. **Derive before coding.** Every method is worked out by hand -- weak forms, shape functions, sensitivity analysis -- before a line of code is written. The math is the point; the code is the translation.
2. **Validate relentlessly.** Each phase is checked against an analytical solution or a published benchmark. Nothing is trusted until it is verified.
3. **Build in layers.** 1D before 2D, linear before nonlinear, direct solvers before ML acceleration. Each layer rests on a foundation that has already been proven correct.

---

## Roadmap

| Phase | Topic | Status |
|:-----:|:------|:------:|
| 1 | 1D linear FEA (bar) | Complete |
| 2 | 2D linear elastic FEA (CST) | Complete |
| 3 | Sensitivity analysis (adjoint method) | Planned |
| 4 | SIMP topology optimization | Planned |
| 5 | Manufacturing-constrained generative design | Planned |
| 6 | ML acceleration (neural reparameterization / PINN surrogates) | Planned |

---

## Phase 1: 1D Bar FEA

A complete linear finite element solver for the 1D bar, built entirely from first principles.

### What it does

Solves the governing equation for axial deformation of a bar:
```
d/dx(EA * du/dx) + b(x) = 0,    u(0) = 0,    EA * du/dx|_{x=L} = P
```

The pipeline: build mesh -> element stiffness matrices -> global assembly -> consistent load vector -> apply boundary conditions -> solve `Ku = F` -> post-process.

### Implemented from scratch

- **Element stiffness matrix** -- derived from the weak form and Galerkin discretization with linear (tent) shape functions.
- **Global assembly** -- scattering element matrices into the global stiffness matrix via a connectivity table.
- **Consistent load vector** -- distributed body loads and point loads, derived from the shape-function projection.
- **Dirichlet boundary conditions** -- applied by direct elimination of constrained degrees of freedom.

### Validation

The solver is checked against the closed-form analytical solution. At the nodes, the FEA result matches to **machine precision** (~1e-15) -- a consequence of the nodal superconvergence property of linear elements for this problem.

![Displacement: FEA vs analytical](phase1-1d-bar/displacement.png)

### Convergence study

A mesh-refinement study confirms the method converges at the theoretically predicted rate. For linear elements, the interpolation error scales as **O(h^2)** -- halving the element size quarters the error. On a log-log plot this appears as a straight line of slope 2, and the measured slope is **2.00**.

![Mesh convergence](phase1-1d-bar/convergence.png)

### Running Phase 1

```bash
cd phase1-1d-bar
python run_example.py     # solve the example problem and validate against analytical
python convergence.py     # run the mesh convergence study
```

---

## Phase 2: 2D Linear Elastic FEA (Constant-Strain Triangle)

A from-scratch 2D plane-stress finite element solver using constant-strain triangle (CST) elements. Generalizes the Phase 1 bar solver from a scalar 1-DOF-per-node problem to a vector 2-DOF-per-node elasticity problem.

### What it does

Solves 2D linear elasticity under plane-stress assumptions on a triangulated domain. The element stiffness is the standard
```
k = Bᵀ D B · t · A
```
where B is the 3×6 strain-displacement matrix (built from the CST shape-function gradients), D is the 3×3 plane-stress constitutive matrix, t is thickness, and A is the triangle area. The pipeline mirrors Phase 1: mesh -> element stiffness -> global assembly (2 DOFs/node) -> load vector -> boundary conditions -> solve -> post-process.

### Implemented from scratch

- **Plane-stress constitutive matrix D** -- relating the three stresses to the three strains, including Poisson coupling.
- **CST element stiffness** -- derived from the linear shape functions over a triangle; validated against four physical properties (symmetry, rank-3 deficiency for the three rigid-body modes, zero-energy rigid translation, positive semi-definiteness).
- **Global assembly** -- 2-DOF-per-node scatter via a connectivity table, generalizing the Phase 1 assembly. Validated by confirming the assembled global matrix retains exactly 3 rigid-body modes.
- **Boundary conditions and loads** -- DOF-level Dirichlet constraints (allowing per-direction fixing) and distributed edge loads.

### Validation: cantilever benchmark

The solver is tested on a clamped cantilever (length = 5× depth) under a distributed tip load, compared against Euler-Bernoulli beam theory (δ = PL³/3EI).

![Deformed cantilever](phase2-2d-elasticity/deformed.png)

Constant strain cannot represent the linear bending-strain gradient through the depth of the beam, so when the element bends, it is forced to develop a spurious shear strain that shouldn't be there. That fake shear soaks up energy, and absorbing energy is what makes the element artificially stiff. The result is a coarse mesh that under-predicts deflection — it reports the beam as stiffer than it really is.
Refinement fixes this. A staircase of many small constant-strain triangles, each holding a slightly different constant value, approximates the smooth linear gradient — and the finer the mesh, the finer the staircase, so less spurious shear is generated and the deflection climbs toward the true value.
The converged value lands slightly above the Euler-Bernoulli line because beam theory assumes cross-sections stay perpendicular to the beam axis and ignores shear deformation entirely. Our 2D elasticity model includes it, which makes the true beam slightly more flexible. So the converged FEA settling a few percent above the beam-theory reference isn't error — it's real physics that beam theory leaves out.


### Convergence study

Unlike Phase 1, there is no exact answer to validate against, because Euler-Bernoulli is an idealization, not the true 2D solution. So we use two complementary views to show convergence.

**Plateau view:** This plots tip deflection against mesh refinement. The deflection climbs from ~0.28 (coarsest mesh, heavily shear-locked) to ~0.512 (converged), rising through and just past the Euler-Bernoulli reference of 0.500. That rise-and-cross shape is precisely the shear-locking-then-shear-deformation story: a coarse mesh is artificially stiff, refinement relieves the locking, and the converged value settles slightly above beam theory because 2D elasticity captures shear deformation that Euler-Bernoulli ignores.

| Mesh (nx×ny) | Elements | Tip deflection |
|:---:|:---:|:---:|
| 10×2 | 40 | 0.280 |
| 20×4 | 160 | 0.423 |
| 40×8 | 640 | 0.487 |
| 80×16 | 2560 | 0.507 |
| 160×32 | 10240 | 0.512 |

Euler-Bernoulli reference: **0.500**

**Self-convergence view:** With no analytical truth available, the finest mesh is used as a stand-in reference, and the error of each coarser mesh is measured against it. On a log-log plot the error falls along a clean line, running roughly parallel to the slope-2 reference — a touch steeper, measured at 1.82. It isn't exactly 2.00 because the finest mesh is itself still slightly converging, so it's an imperfect reference, and that contamination pulls the slope down a little. Even so, it confirms the second-order-ish convergence expected for CST displacement.

**Together:** the plateau confirms *what* the solver converges to (the physics), and the self-convergence confirms *how fast* (the numerics). Together they validate the solver without an exact answer — the kind of verification you'd use on any real part where no closed-form solution exists.

![Plateau convergence](phase2-2d-elasticity/convergence_plateau.png)
![Self-convergence](phase2-2d-elasticity/convergence_selfconv.png)
### Running Phase 2

```bash
cd phase2-2d-elasticity
python run_example.py     # solve the cantilever, save deformed mesh
python convergence.py     # plateau + self-convergence study
```

---

## Tech Stack

- **Python 3.11**
- **NumPy** -- array math, linear algebra
- **Matplotlib** -- visualization

(Later phases will add SciPy, PyTorch, and optimization libraries.)

---

## License

MIT -- see [LICENSE](LICENSE).