# Mesh movement using the lineal spring approach
# ==============================================
#
# In this demo, we demonstrate a basic example using the *lineal spring* method, as
# described in :cite:`Farhat:1998`.
#
# The idea of the lineal spring method is to re-interpret the edges of a mesh as a
# structure of stiff beams. Each beam has a stiffness associated with it, which is
# related to its length and its orientation. We can assemble this information as a
# *stiffness matrix*,
#
# .. math::
#     \underline{\mathbf{K}} = \begin{bmatrix}
#     \underline{\mathbf{K}_{11}} && \dots && \underline{\mathbf{K}_{1N}}\\
#     \vdots && \ddots && \vdots\\
#     \underline{\mathbf{K}_{N1}} && \dots && \underline{\mathbf{K}_{NN}}\\
#     \end{bmatrix},
#
# where :math:`N` is the number of vertices in the mesh and each block
# :math:`\underline{\mathbf{K}_{ij}}` is a zero matrix if and only if vertex :math:`i`
# is *not* connected to vertex :math:`j`. For a 2D problem, each
# :math:`\underline{\mathbf{K}_{ij}}\in\mathbb{R}^{2\times2}` and
# :math:`\underline{\mathbf{K}}\in\mathbb{R}^{2N\times2N}`.
#
# Suppose we apply a forcing, which acts on the vertices according to a forcing matrix,
#
# .. math::
#     \underline{\mathbf{f}} = \begin{bmatrix}
#         \mathbf{f}_1\\
#         \vdots\\
#         \mathbf{f}_N\\
#     \end{bmatrix}
#     \in\mathbb{R}^{N\times2}.
#
# That is, vertex :math:`i` is forced according to a vector :math:`\mathbf{f}_i`. Then
# we are able to compute the displacement of the vertices by solving the linear system
#
# .. math::
#     \underline{\mathbf{K}} \mathbf{u} = \mathbf{f},
#
# where :math:`\mathbf{u}\in\mathbb{R}^{2N}` and :math:`\mathbf{f}\in\mathbb{R}^{2N}`
# are 'flattened' versions of the displacement and forcing vectors. By solving this
# equation, we see how the structure of beams responds to the forcing.
#
# We begin by importing from the namespaces of Firedrake and Movement.

from firedrake import *
from movement import *

# Let's start with a uniform mesh of the unit square. It has four boundary segments,
# which are tagged with the integers 1, 2, 3, and 4. Note that segment 4 corresponds to
# the top boundary.

import matplotlib.pyplot as plt
from firedrake.pyplot import triplot

n = 10
mesh = UnitSquareMesh(n, n)
fig, axes = plt.subplots()
triplot(mesh, axes=axes)
axes.set_aspect(1)
axes.legend()
plt.savefig("lineal_spring-initial_mesh.jpg")

# .. figure:: lineal_spring-initial_mesh.jpg
#    :figwidth: 60%
#    :align: center
#
# Suppose we wish to apply a time-dependent forcing to the top boundary and see how the
# mesh structure responds. Consider the forcing
#
# .. math::
#     \mathbf{f}(x,y,t)=\left[0, A\:\sin\left(\frac{2\pi t}T\right)\:\sin(\pi x)\right]
#
# acting only in the vertical direction.

import numpy as np

A = 0.5  # forcing amplitude
T = 1.0  # forcing period


def forcing(x, t):
    return A * np.sin(2 * pi * t / T) * np.sin(pi * x)


X = np.linspace(0, 1, n + 1)
dt = 0.1
times = np.arange(0, 1.001, dt)

fig, axes = plt.subplots()
for t in times:
    axes.plot(X, forcing(X, t), label=f"{t:.1f}")
axes.set_xlim([0, 1])
axes.set_ylim([-A, A])
axes.legend()
plt.savefig("lineal_spring-forcings.jpg")

# .. figure:: lineal_spring-forcings.jpg
#    :figwidth: 60%
#    :align: center
#
# To apply this forcing, we need to create a :class:`~.SpringMover` instance and define
# a function for updating the forcing applied to the boundary nodes. The way we get the
# right indices for the top boundary is using a :class:`~.DirichletBC` object.

mover = SpringMover(mesh, method="lineal")
V = mesh.coordinates.function_space()
boundary_nodes = DirichletBC(V, 0, 4).nodes


def update_forcings(t):
    coords = mover.mesh.coordinates.dat.data
    for i in boundary_nodes:
        mover.f.dat.data[i, 1] = forcing(coords[i, 0], t)


# We are now able to apply the mesh movement method. The forcings effectively enforce a
# Dirichlet condition on the top boundary. On other boundaries, we enforce that there is
# no movement using the `fixed_boundaries` keyword argument.

results = []
for t in times:
    mover.move(t, update_forcings=update_forcings, fixed_boundaries=[1, 2, 3])
    print(
        f"time = {t:.1f} s, displacement = {np.linalg.norm(mover.displacement):.2f} m"
    )
    results.append(Mesh(mover.mesh.coordinates.copy(deepcopy=True)))

# TODO: Plot grid of adapted meshes

# Note that we can view the sparsity pattern of the stiffness matrix as follows.

K = mover.stiffness_matrix
print(f"Stiffness matrix shape: {K.shape}")
print(f"Number of mesh vertices: {mesh.num_vertices()}")

fig, axes = plt.subplots()
axes.spy(K)
plt.savefig("lineal_spring-sparsity.jpg")

# This tutorial can be dowloaded as a `Python script <lineal_spring.py>`__.
#
#
# .. rubric:: References
#
# .. bibliography:: demo_references.bib
#    :filter: docname in docnames
