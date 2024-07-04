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
# the computational and physical domains, respectively. Skipping some of the details
# that can be found in :cite:`McRae:2018`, of the possible mappings we choose one that
# takes the form
#
# .. math::
#     \mathbf{x} = \boldsymbol{\xi}+\nabla_{\boldsymbol{\xi}}\phi,
#
# where :math:`\phi` is a convex potential function. Further, we choose the potential
# such that it is the solution of the Monge-Ampère type equation,
#
# .. math::
#     m(\mathbf{x}) \det(\underline{\mathbf{I}}+\nabla_{\boldsymbol{\xi}}\nabla_{\boldsymbol{\xi}}\phi) = \theta,
#
# where :math:`m(\mathbf{x})` is a so-called *monitor function* and :math:`\theta` is a
# strictly positive normalisation function. The monitor function is of key importance
# because it specifies the desired *density* of the adapted mesh across the domain,
# i.e., where resolution is focused. Note that density is the reciprocal of area in 2D
# or of volume in 3D.
#
# We begin the example by importing from the namespaces of Firedrake and Movement. ::

import os

from firedrake import *

from movement import *

# To start with a simple example, consider a uniform mesh of the unit square. Feel free
# to ignore the `"MOVEMENT_REGRESSION_TEST"`, as it is only used when this demo is run
# in the test suite (to reduce its runtime). ::

test = os.environ.get("MOVEMENT_REGRESSION_TEST")
n = 10 if test else 20
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
# :class:`~movement.monge_ampere.MongeAmpereMover` instance and adapt the mesh. By default, the Monge-Ampère
# equation is solved to a relative tolerance of :math:`10^{-8}`. However, for the
# purposes of continuous integration testing, a tolerance of :math:`10^{-3}` is used
# instead to further reduce the runtime. ::

rtol = 1.0e-03 if test else 1.0e-08
mover = MongeAmpereMover(mesh, ring_monitor, method="quasi_newton", rtol=rtol)
mover.move()

# This should give command line output similar to the following:
#
# .. code-block:: none
#
#       0   Volume ratio 11.49   Variation (σ/μ) 9.71e-01   Residual 9.39e-01
#       1   Volume ratio  8.32   Variation (σ/μ) 6.84e-01   Residual 5.35e-01
#       2   Volume ratio  5.74   Variation (σ/μ) 5.55e-01   Residual 3.83e-01
#       3   Volume ratio  6.86   Variation (σ/μ) 4.92e-01   Residual 3.06e-01
#       4   Volume ratio  5.91   Variation (σ/μ) 4.53e-01   Residual 2.69e-01
#       5   Volume ratio  8.38   Variation (σ/μ) 4.20e-01   Residual 2.22e-01
#       6   Volume ratio  7.34   Variation (σ/μ) 4.12e-01   Residual 2.14e-01
#       7   Volume ratio  7.68   Variation (σ/μ) 4.02e-01   Residual 2.03e-01
#       8   Volume ratio  7.93   Variation (σ/μ) 3.84e-01   Residual 1.84e-01
#       9   Volume ratio  7.81   Variation (σ/μ) 3.83e-01   Residual 1.86e-01
#      10   Volume ratio  7.60   Variation (σ/μ) 3.93e-01   Residual 1.97e-01
#      11   Volume ratio  7.99   Variation (σ/μ) 4.14e-01   Residual 2.13e-01
#      12   Volume ratio  8.22   Variation (σ/μ) 4.21e-01   Residual 2.20e-01
#      13   Volume ratio 10.79   Variation (σ/μ) 4.54e-01   Residual 2.13e-01
#      14   Volume ratio  9.66   Variation (σ/μ) 4.15e-01   Residual 1.33e-01
#      15   Volume ratio 10.52   Variation (σ/μ) 3.75e-01   Residual 9.77e-02
#      16   Volume ratio 10.00   Variation (σ/μ) 3.90e-01   Residual 8.64e-02
#      17   Volume ratio  9.00   Variation (σ/μ) 3.61e-01   Residual 6.33e-02
#      18   Volume ratio  9.53   Variation (σ/μ) 3.73e-01   Residual 4.41e-02
#      19   Volume ratio  8.86   Variation (σ/μ) 3.60e-01   Residual 3.71e-02
#      20   Volume ratio  9.38   Variation (σ/μ) 3.65e-01   Residual 2.71e-02
#      21   Volume ratio  8.95   Variation (σ/μ) 3.57e-01   Residual 2.23e-02
#      22   Volume ratio  9.15   Variation (σ/μ) 3.57e-01   Residual 1.32e-02
#      23   Volume ratio  8.90   Variation (σ/μ) 3.52e-01   Residual 8.93e-03
#      24   Volume ratio  8.87   Variation (σ/μ) 3.50e-01   Residual 3.93e-03
#      25   Volume ratio  8.80   Variation (σ/μ) 3.48e-01   Residual 2.61e-03
#      26   Volume ratio  8.85   Variation (σ/μ) 3.49e-01   Residual 1.51e-03
#      27   Volume ratio  8.83   Variation (σ/μ) 3.48e-01   Residual 1.15e-03
#      28   Volume ratio  8.85   Variation (σ/μ) 3.48e-01   Residual 7.98e-04
#      29   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 6.27e-04
#      30   Volume ratio  8.85   Variation (σ/μ) 3.48e-01   Residual 4.46e-04
#      31   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 3.46e-04
#      32   Volume ratio  8.85   Variation (σ/μ) 3.48e-01   Residual 2.39e-04
#      33   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 1.77e-04
#      34   Volume ratio  8.85   Variation (σ/μ) 3.48e-01   Residual 1.14e-04
#      35   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 7.82e-05
#      36   Volume ratio  8.85   Variation (σ/μ) 3.48e-01   Residual 4.69e-05
#      37   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 2.96e-05
#      38   Volume ratio  8.85   Variation (σ/μ) 3.48e-01   Residual 1.77e-05
#      39   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 1.11e-05
#      40   Volume ratio  8.85   Variation (σ/μ) 3.48e-01   Residual 7.43e-06
#      41   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 5.07e-06
#      42   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 3.86e-06
#      43   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 2.85e-06
#      44   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 2.30e-06
#      45   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 1.72e-06
#      46   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 1.38e-06
#      47   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 9.75e-07
#      48   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 7.42e-07
#      49   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 4.50e-07
#      50   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 3.00e-07
#      51   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 1.42e-07
#      52   Volume ratio  8.84   Variation (σ/μ) 3.48e-01   Residual 7.93e-08
#    Solver converged in 52 iterations.
#
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
# Whilst it looks like the mesh might have tangled elements, closer inspection shows
# that this is not the case.

fig, axes = plt.subplots()
triplot(mover.mesh, axes=axes)
axes.set_xlim([0.15, 0.3])
axes.set_ylim([0.15, 0.3])
axes.set_aspect(1)
plt.savefig("monge_ampere1-adapted_mesh_zoom.jpg")

# .. figure:: monge_ampere1-adapted_mesh_zoom.jpg
#    :figwidth: 60%
#    :align: center
#
# .. rubric:: Exercise
#
# To further convince yourself that there are no tangled elements, go back to the start
# of the demo and set up a :class:`movement.tangling.MeshTanglingChecker` object using
# the initial mesh. Use it to check for tangling after the mesh movement has been
# applied.
#
# In the `next demo <./monge_ampere_3d.py.html>`__, we will demonstrate
# that the Monge-Ampère method can also be applied in three dimensions.
#
# This tutorial can be dowloaded as a `Python script <monge_ampere1.py>`__.
#
#
# .. rubric:: References
#
# .. bibliography:: demo_references.bib
#    :filter: docname in docnames
