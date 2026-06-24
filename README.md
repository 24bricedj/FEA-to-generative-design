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

## Capstone Target

The project builds toward a concrete deliverable: a **topology-optimized, internally-cooled, additively-manufactured piston** for a high-output engine. This part unifies every layer of the stack — structural loading (combustion gas pressure and reciprocating inertia), thermal loading (combustion heat flux and internal oil-cooling galleries), and a geometry that can only be produced by additive manufacturing. It's the reason the roadmap includes a from-scratch thermal solver: a piston is a coupled thermal-structural problem, and optimizing one honestly requires capturing both fields.

The broader thesis is unchanged — build every layer from first principles to understand what commercial generative-design and AM tools do internally — with the application target sharpened toward propulsion and powertrain performance.

---


## Roadmap

| Phase | Topic | Status |
|:-----:|:------|:------:|
| 1 | 1D linear FEA (bar) | Complete |
| 2 | 2D linear elastic FEA (CST) | Complete |
| 3 | Thermal FEA (1D → 2D conduction → thermal-structural coupling) | Complete |
| 4 | Sensitivity analysis (adjoint method) | Planned |
| 5 | SIMP topology optimization | Planned |
| 6 | Manufacturing-constrained generative design (AM overhang, feature size) | Planned |
| 7 | ML acceleration (neural reparameterization / PINN surrogates) | Planned |

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

## Phase 3: Thermal FEA & Thermal-Structural Coupling

A from-scratch steady-state heat-conduction solver -- 1D, then 2D on the CST mesh -- culminating in **thermal-structural coupling**: joining the thermal and elasticity solvers so a temperature field produces real thermal stress and deformation. This is the multiphysics capability the piston capstone is built on.

### The core insight: thermal conduction ≡ elasticity

1D steady-state heat conduction is a direct isomorphism with the Phase 1 elasticity bar. Starting from the same weak form, you simply substitute each elasticity quantity for its thermal counterpart -- displacement → temperature, Young's modulus → conductivity, body load → heat source, internal force → heat flow rate (Fourier's law) -- and the derivation follows the identical path. The problem becomes almost exactly the one already solved in Phase 1, so the existing solver *became* the 1D thermal solver by recognizing the structure rather than rewriting it. With constant coefficients and no spatial variation in the weak-form terms, the derivation stays simple.

The one genuinely new piece versus Phase 1 is **nonzero Dirichlet BCs**. Phase 1 only ever fixed values to zero, where the constrained term contributes nothing and drops out of the system. Here, a prescribed nonzero end temperature *does* contribute: its known value, scaled by the coupling stiffness, injects a load into every neighboring equation (`F_i -= K_ij · T_j`) before the fixed row and column are eliminated. Phase 1 is the special case of this where the prescribed value is zero, so the injected load vanishes.

### Step 1: 1D heat conduction

Solves steady-state conduction in a bar:
```
d/dx(kA dT/dx) + Q = 0
```

- **Element conductivity matrix** -- `(kA/h)[[1,-1],[-1,1]]`, the Phase 1 stiffness matrix under the isomorphism (E → k).
- **Nonzero Dirichlet BCs** -- the one genuinely new piece vs. Phase 1, derived by hand and validated.
- **Validation** -- against two hand-derived analytical profiles: a linear profile (no source) and a quadratic profile (uniform source), both matching to machine precision (~1e-15) at the nodes. Convergence confirmed at **O(h²)**, slope 2.00.

![1D thermal: source-driven quadratic profile](phase3-thermal/1d-conduction/profile_caseB.png)

### Step 2: 2D heat conduction (CST)

Generalizes to 2D on the same triangular mesh as Phase 2. Because temperature is a **scalar** (1 DOF/node) where displacement was a vector (2 DOF/node), the element is simpler than Phase 2: a **3×3** conductivity matrix built from a **2×3** temperature-gradient matrix, with scalar conductivity replacing the 3×3 constitutive matrix.

```
k_e = Bᵀ B · k · A,    B = (1/2A) [[b1,b2,b3],[c1,c2,c3]]
```

- **Insulated boundaries for free** -- a zero-flux edge is a natural (Neumann) BC that falls out of the weak form automatically. Insulated edges require *no code*: doing nothing to a boundary enforces zero heat flux through it.
- **Validation** -- a linear-profile benchmark (insulated top/bottom, fixed sides) matches to machine precision, confirming the insulated-edge behavior. A curved **sinusoidal-edge benchmark** (exact sin·sinh solution) gives **O(h²)** convergence, slope 1.975.

![2D temperature field](phase3-thermal/2d-conduction/temperature_field.png)

### Step 3: Thermal-structural coupling

A heated material wants to expand, which creates thermal strain. If the material is free, it expands as much as it wants and develops no stress. But if it is constrained -- physically pinned, bolted, or held back by cooler neighboring material -- that frustrated expansion produces stress. The same holds in reverse for thermal contraction under cooling. The key is that total strain splits into a mechanical part and a thermal part, and only the mechanical part -- the gap between what the material actually does and what it freely wanted to do -- produces stress: σ = D(ε_total − ε_thermal).

The bridge between the solvers is that this thermal strain enters the elasticity system as an equivalent load: Ku = F_ext + F_thermal. The temperature field computed by the thermal solver becomes a force driving the structural solver. This is exactly what the piston capstone needs -- a part that heats unevenly, is constrained, and so develops thermal stress on top of the mechanical stress from gas pressure and inertia. In a mission-critical part like a piston, both must be accounted for; thermal stress alone can cause failure.

The coupling is validated against a closed-form case -- a fully-constrained uniformly-heated block, whose exact thermal stress is `σ = -EαΔT/(1-ν)` -- matching to machine precision.

The demo below shows the pure thermal effect isolated: a cantilever with a hot-bottom/cool-top gradient and **no mechanical load** curls upward like a bimetallic strip, driven entirely by differential thermal expansion. (Shown at true scale -- thermal deformation is a large, real effect.)

![Thermal curling under a gradient](phase3-thermal/coupling/thermal_curl.png)

### Running Phase 3

```bash
cd phase3-thermal/1d-conduction
python run_example.py     # 1D validation (linear + quadratic profiles)
python convergence.py     # O(h²) convergence

cd ../2d-conduction
python run_example.py     # 2D sinusoidal benchmark + temperature field
python convergence.py     # O(h²) convergence

cd ../coupling
python run_example.py            # constrained-block thermal stress (validation)
python thermal_curl_demo.py      # thermal curling visualization
```

---

## Tech Stack

- **Python 3.11**
- **NumPy** -- array math, linear algebra
- **Matplotlib** -- visualization

(Later phases will add SciPy, PyTorch, and optimization libraries.)

---
- **NumPy** -- array math, linear algebra
- **Matplotlib** -- visualization

(Later phases will add SciPy, PyTorch, and optimization libraries.)

---

## License

MIT -- see [LICENSE](LICENSE).