---
title: 'Movement: a library of mesh movement methods for Firedrake'
tags:
  - Python
  - mesh adaptation
  - finite element
  - Firedrake
authors:
  - name: Joseph G. Wallwork
    affiliation: "1"
    orcid: 0000-0002-3646-091X
    corresponding: true
  - name: Stephan C. Kramer
    affiliation: "2"
    orcid: 0000-0002-9193-5092
  - name: Davor Dundovic
    affiliation: "3"
    orcid: 0009-0000-6085-4794
  - name: Mingrui Zhang
    affiliation: "2"
    orcid: 0000-0002-9549-5813
  - name: Matthew D. Piggott
    affiliation: "2"
    orcid: 0000-0002-7526-6853
affiliations:
 - name: Institute of Computing for Climate Science, University of Cambridge, UK
   index: 1
 - name: Department of Earth Science and Engineering, Imperial College London, UK
   index: 2
 - name: Department of Geosciences, University of Oslo, Norway
   index: 3
date: 4 September 2024
bibliography: paper.bib

---

# Summary

*Movement* is a library of mesh movement methods for the Firedrake finite
element package [@Firedrake]. It implements several commonly used mesh movement
approaches from the literature.

*Mesh adaptation* is an advanced discretisation approach for mesh-based partial
differential equation (PDE) solvers. It uses variable spatial resolution, guided
by a user-specified error indicator or heuristic, in an attempt to improve
accuracy, whilst maintaining a low overall computational cost. *Mesh movement*
is family of methods in which the topology of the mesh is unaffected by the mesh
adaptation process. That is, mesh entities (vertices, edges, cells, etc.) are
not inserted or deleted and the connectivity of the mesh remains unchanged. What
does change under adaptation is the *geometry* of the mesh - namely the mesh
vertex positions. This affects the spatial resolution because vertices (and
other mesh entities) are re-distributed, leaving some regions with higher
resolution than others. Mesh movement is also known as $r$-adaptation, where the
$r$ stands for *re-distribution*.

# Statement of need

Several works have implemented mesh movement in Firedrake, such as
[@McManus17;@McRae18;@Paganini21;@Clare22]. These implementations include
several different mesh movement methods, as described subsequently. However,
there does not currently exist a single, up-to-date package providing a
user-friendly 'toolbox' of various different mesh movement methods.

# Software description

## Underlying framework

Movement is a Python package that uses the domain-specific languages of
Firedrake [@Firedrake] and Unified Form Language (UFL) [@UFL]. With these, the
user is able to write high-level code for finite element problems, akin to how
finite element notation is written as mathematics. Despite the high-level
interface, Firedrake achieves efficiency through automatic code generation and
by leveraging PETSc [@PETSc] for its unstructured mesh representation and
linear and nonlinear solvers. The mesh movement methods implemented in Movement
are PDE-based, meaning it inherits these efficiency benefits from Firedrake.

Movement is an open source package with an MIT licence. The source code is
freely available at <https://github.com/mesh-adaptation/movement>. Long-form
documentation, API documentation, and demos may be found at
<https://mesh-adaptation.github.io/docs/movement/index.html> - part of a larger
website, which also provides documentation for other mesh adaptation packages
developed and maintained by our group. Movement has an open development process
and welcomes contributors. Development guidelines can be found within the wiki
pages for the wider `mesh-adaptation` organisation at
<https://github.com/mesh-adaptation/mesh-adaptation-docs/wiki/Development-Practices.>

## Process

Firedrake is an essential dependency of Movement, so it should be installed
first, following the instructions at <https://firedrakeproject.org/download>.
Having done so, activate the virtual environment and pip install either by
cloning the Movement repository and running `pip install ./movement` or pip
installing directly from GitHub with
```sh
pip install git+https://github.com/mesh-adaptation/movement.git
```

Movement is provided as a Python module, which can be imported with
```python
from movement import *
```
or similar. Assuming that the repository was cloned locally, the demos may be
run by navigating to the `demos` subdirectory and running the Python scripts
from the command line.

## Main library

Movement takes an object-oriented approach, focused on `Mover` classes, which
handle the transformation of an input mesh into an adapted mesh. There
are three main families of `Mover`s, aligned with the three main mesh movement
paradigms.

<!--
For many mesh movement methods, we make use of two domain concepts:
the *computational domain* $\Omega_C$, which is unchanged during the
computation, and the *physical domain* $\Omega_P$, which we seek to use outside
of the mesh movement process, for example to solve PDEs in. Associated with
these are a *computational mesh* $\mathcal{H}_C$ and *physical mesh*
$\mathcal{H}_P$, respectively. As such, the mesh movement process may be
described as the search for a continuous map $\Omega_C\rightarrow\Omega_P$. In
practice, we search for a map between the discrete meshes
$\mathcal{H}_C\rightarrow\mathcal{H}_P$.
-->

### Monitor-based methods

These methods use the concept of a *monitor function* - a strictly positive,
scalar function which is sought to be *equi-distributed* over the physical
domain [@Budd09]. In practice, this means that the integral of the monitor
function is approximately equal across the elements of the physical mesh. As
such, the monitor function is often thought of as a 'mesh density function'.

The two monitor-based methods currently implemented in Movement are based on
optimal transport theory and are driven by solutions of Monge-Ampère type
equations. The idea is to solve auxiliary nonlinear PDEs to determine the
minimal mesh deformation that will equidistribute a user-provided monitor
function. The implementations of both Monge-Ampère methods are based on those
used to generate the results presented in [@McRae18].

Figure 1 shows an example from the first Monge-Ampère based Movement demo, with
an analytically prescribed monitor function,
\begin{equation}\label{eq:ring}
  m(x,y) = 1 + \frac{\alpha}{\cosh^2\left(\beta\left(
    \left(x-\frac{1}{2}\right)^2+\left(y-\frac{1}{2}\right)^2-\gamma^2\right)
  \right)},
\end{equation}
with amplitude $\alpha=20$, width $\beta=200$, and radius $\gamma^2=0.15$.

![Monge-Ampère based mesh movement applied to a uniform mesh, with ring-shaped
monitor function (\ref{eq:ring}). Left: full
mesh. Right: zoom on the region $[0.15,0.3]\times[0.15,0.3]$.](ma_demo.png)

### Velocity-based methods

Velocity-based methods introduce the concept of a *mesh velocity*, which is
used to update the mesh between iterations of a time integration scheme for
time-dependent PDE problems [@McManus17]. They can also be used to support PDE
solvers with moving reference frames, such as Laplacian or Arbitrary
Lagrangian-Eulerian (ALE) viewpoints.

Movement implements Laplacian smoothing [@Field88] as a velocity-based method.
This approach is now standard in the mesh adaptation literature, although it is
generally used as a post-processing step for improving the quality of a mesh
generated by a $h$-adaptive (topology-deforming) method [@McManus17]. Given some
forcing of the mesh, the Laplacian smoothing method determines the mesh
response by solving a vector Laplace equation as an auxiliary PDE.

### Spring-based methods

The spring-based approach differs from the others in that it
re-interprets the mesh as a fictitious, discrete structure comprised of beams.
Each beam is given a stiffness value and the mesh response to forcings is
determined by solving a discrete linear elasticity problem. Movement currently
only implements the *lineal spring* approach, although there are plans to
implement the *torsional spring* approach [@Farhat98].

## Tools

In addition to the core functionality of the mesh movement methods described
above, Movement implements supporting utilities including:

- The `MeshTanglingChecker` class, which can be used to detect when one or more
  mesh elements has become invalid as a result of the mesh movement process.
  Such a checker is created and enabled by default along with any `Mover`
  instance.
- Builder classes for creating standard monitor functions, including analytical
  shapes such as rings and balls, as well as monitors based on gradients and/or
  Hessians of solution PDE fields.

<!--
## Comparison to other approaches

[TODO?]
-->

# Examples of use

Movement has been used to implement mesh movement methods in several research
projects. To the best of our knowledge, only the Monge-Ampère based methods
have so far been utilised. Research papers that make use of Movement include:

- [@dSO24;@dSO25]: Monge-Ampère based mesh movement for seismology applications.
- [@M2N]: End-to-end graph neural network (GNN) emulators for mesh movement.
- [@UM2N]: Extension of [@M2N] that generalises to arbitrary PDEs.
- [@RM24]: A different GNN mesh movement framework that makes use of online
  training via Firedrake's in-built automatic differentiation functionality.

Note that [@dSO24] uses Movement directly within an application code, whereas
[@M2N;@UM2N;@RM24] use Movement's Monge-Ampère `Mover`s to provide
'gold standard' adapted meshes that are used as training data for GNNs.

# Future development

As described above, Movement currently implements one velocity-based method, one
spring-based method, and one monitor-based method (albeit with two different
solution strategies). Rather than implementing more of the many methods
presenting in the literature, development has focused on improving the
functionality, efficiency, and robustness of the Monge-Ampère `Mover`s, since
these are the best used by collaborators and users.

Now that the Monge-Ampère implementations have reached a mature state, we seek
to implement several other methods, including *torsional spring* [@Farhat98],
*linear and nonlinear elasticity*, *MMPDE methods* [@Huang94] *parabolic
Monge-Ampère*, [@Budd06], and anisotropic variants of various monitor-based
approaches [@Huang05]. We also plan to implement tools for integrating the
`Mover`s into Lagrangian and ALE solution strategies for PDE solvers.

# Acknowledgments

The authors gratefully acknowledge funding from Huawei Corporation Ltd.

# References
