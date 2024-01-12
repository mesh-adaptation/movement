# Introduction to mesh movement driven by a Monge-Ampère type equation
# ====================================================================

# In this demo, we consider an example of mesh movement driven by solutions of a
# Monge-Ampère type equation.
#
# The idea is to consider two domains: the *physical domain* :math:`\Omega_P` and the
# *computational domain* :math:`\Omega_C`. Associated with these are the *physical mesh*
# :math:`\mathcal{H}_P` and the *computational mesh* :math:`\mathcal{H}_C`. The
# computational domain and mesh remain fixed throughout the simulation, whereas the
# physical domain and mesh are allowed to change. In this framework, we can interpret
# mesh movement algorithms as searches for mappings between the computational and
# physical domains:
#
# .. math::
#     \mathbf{x}:\Omega_C\rightarrow\Omega_P.
#
# In practice we are really searching for a discrete mapping between the computational
# and physical meshes.
#
# Let :math:`\boldsymbol{\xi}` and :math:`\mathbf{x}` denote the coordinate fields in
# the computational and physical domains. Skipping some of the details that can be
# found in :cite:`McRae:2018`, of the possible mappings we choose one that takes the form
#
# .. math::
#     \mathbf{x} = \boldsymbol{\xi}+\nabla_{\boldsymbol{\xi}}\phi,
#
# where :math:`\phi` is a convex potential function. Further, we choose the potential
# such that is the solution of the Monge-Ampère type equation,
#
# .. math::
#     m(\mathbf{x}) \det(\underline{\mathbf{I}}+\nabla_{\boldsymbol{\xi}}\nabla_{\boldsymbol{\xi}}\phi) = \theta,
#
# where :math:`m(\mathbf{x})` is a so-called *monitor function* and :math:`\theta` is a
# strictly positive normalisation function. The monitor function is of key importance
# because it specifies the desired *density* of the adapted mesh across the domain,
# i.e., where resolution is focused. (Note that density is the reciprocal of area in 2D
# or of volume in 3D.)
#
# We begin the example by importing from the namespaces of Firedrake and Movement.

from firedrake import *
from movement import *

# To start with a simple example, consider a uniform mesh of the unit square.

n = 20
mesh = UnitSquareMesh(n, n)

# We can plot the initial mesh using Matplotlib as follows.

import matplotlib.pyplot as plt
from firedrake.pyplot import triplot

fig, axes = plt.subplots()
triplot(mesh, axes=axes)
axes.set_aspect(1)
plt.savefig("monge_ampere1-initial_mesh.jpg")

# .. figure:: monge_ampere1-initial_mesh.jpg
#    :figwidth: 60%
#    :align: center
#
# Let's choose a monitor function which focuses mesh density in a ring centred within
# the domain, according to the formula,
#
# .. math::
#     m(x,y) = 1 + \frac{\alpha}{\cosh^2\left(\beta\left(\left(x-\frac{1}{2}\right)^2+\left(y-\frac{1}{2}\right)^2-\gamma\right)\right)},
#
# for some values of the parameters :math:`\alpha`, :math:`\beta`, and :math:`\gamma`.
# Unity is added at the start to ensure that the monitor function doesn't get too
# close to zero.
#
# Here we can think of :math:`\alpha` as relating to the amplitude of the monitor
# function, :math:`\beta` as relating to the width of the ring, and :math:`\gamma` as
# the radius of the ring.


def ring_monitor(mesh):
    alpha = Constant(20.0)
    beta = Constant(200.0)
    gamma = Constant(0.15)
    x, y = SpatialCoordinate(mesh)
    r = (x - 0.5) ** 2 + (y - 0.5) ** 2
    return Constant(1.0) + alpha / cosh(beta * (r - gamma)) ** 2


# With an initial mesh and a monitor function, we are able to construct a
# :class:`~.MongeAmpereMover` instance and adapt the mesh.

mover = MongeAmpereMover(mesh, ring_monitor, method="quasi_newton")
mover.move()

# The adapted mesh can be accessed via the `mesh` attribute of the mover. Plotting it,
# we see that the adapted mesh has its resolution focused around a ring, as expected.

fig, axes = plt.subplots()
triplot(mover.mesh, axes=axes)
axes.set_aspect(1)
plt.savefig("monge_ampere1-adapted_mesh.jpg")

# .. figure:: monge_ampere1-adapted_mesh.jpg
#    :figwidth: 60%
#    :align: center
#
# This tutorial can be dowloaded as a `Python script <monge_ampere1.py>`__.
#
#
# .. rubric:: References
#
# .. bibliography:: demo_references.bib
#    :filter: docname in docnames
