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
#     m(\mathbf{x}) \det(\underline{\mathbf{I}}
#     +\nabla_{\boldsymbol{\xi}}\nabla_{\boldsymbol{\xi}}\phi) = \theta,
#
# where :math:`m(\mathbf{x})` is a so-called *monitor function* and :math:`\theta` is a
# strictly positive normalisation function. The monitor function is of key importance
# because it specifies the desired *density* of the adapted mesh across the domain,
# i.e., where resolution is focused. Note that density is the reciprocal of area in 2D
# or of volume in 3D.
#
# We begin the example by importing from the namespaces of Firedrake and Movement. ::

from firedrake import *

from movement import *

# To start with a simple example, consider a uniform mesh of the unit square. ::

n = 20
mesh = UnitSquareMesh(n, n)

# We can plot the initial mesh using Matplotlib as follows.

import matplotlib.pyplot as plt
from firedrake.pyplot import triplot

fig, axes = plt.subplots()
triplot(mesh, axes=axes)
axes.set_aspect(1)
plt.savefig("monge_ampere_ring-initial_mesh.jpg")

# .. figure:: monge_ampere_ring-initial_mesh.jpg
#    :figwidth: 60%
#    :align: center
#
# Let's choose a monitor function which focuses mesh density in a ring centred within
# the domain, according to the formula,
#
# .. math::
#     m(x,y) = 1 + \frac{\alpha}{\cosh^2\left(\beta\left(
#     \left(x-\frac{1}{2}\right)^2+\left(y-\frac{1}{2}\right)^2-\gamma\right)\right)},
#
# for some values of the parameters :math:`\alpha`, :math:`\beta`, and :math:`\gamma`.
# Unity is added at the start to ensure that the monitor function doesn't get too
# close to zero. Here we can think of :math:`\alpha` as relating to the amplitude of the
# monitor function, :math:`\beta` as relating to the width of the ring, and
# :math:`\gamma` as the radius of the ring.
#
# For convenience, Movement provides a builder class for ring monitors,
# :class:`~movement.monitor.RingMonitorBuilder`, amongst other commonly used
# monitor functions. ::

mb = RingMonitorBuilder(centre=(0.5, 0.5), radius=0.4, amplitude=20.0, width=200.0)
ring_monitor = mb.get_monitor()

# With an initial mesh and a monitor function, we are able to construct a
# :class:`~movement.monge_ampere.MongeAmpereMover` instance and adapt the mesh. By
# default, the Monge-Ampère equation is solved to a relative tolerance of
# :math:`10^{-8}`. ::

rtol = 1.0e-08
mover = MongeAmpereMover(mesh, ring_monitor, method="quasi_newton", rtol=rtol)
mover.move()

# This should give command line output similar to the following:
#
# .. code-block:: none
#
#       0   Volume ratio 12.91   Variation (σ/μ) 1.21e+00   Residual 1.16e+00
#       1   Volume ratio  6.77   Variation (σ/μ) 6.15e-01   Residual 4.85e-01
#       2   Volume ratio  5.84   Variation (σ/μ) 5.69e-01   Residual 4.26e-01
#       3   Volume ratio  6.34   Variation (σ/μ) 5.14e-01   Residual 3.74e-01
#       4   Volume ratio  6.38   Variation (σ/μ) 4.88e-01   Residual 3.22e-01
#       5   Volume ratio 12.05   Variation (σ/μ) 4.55e-01   Residual 2.57e-01
#       6   Volume ratio 11.69   Variation (σ/μ) 4.26e-01   Residual 2.30e-01
#       7   Volume ratio 11.98   Variation (σ/μ) 4.23e-01   Residual 1.98e-01
#       8   Volume ratio 11.81   Variation (σ/μ) 4.17e-01   Residual 1.95e-01
#       9   Volume ratio 12.00   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      10   Volume ratio 12.07   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      11   Volume ratio 12.09   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      12   Volume ratio 12.09   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      13   Volume ratio 12.09   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      14   Volume ratio 12.09   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      15   Volume ratio 12.09   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      16   Volume ratio 12.09   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      17   Volume ratio 12.09   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      18   Volume ratio 12.09   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      19   Volume ratio 12.09   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      20   Volume ratio 12.09   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      21   Volume ratio 12.09   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      22   Volume ratio 12.09   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      23   Volume ratio 12.09   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      24   Volume ratio 12.10   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      25   Volume ratio 12.18   Variation (σ/μ) 4.16e-01   Residual 1.94e-01
#      26   Volume ratio 12.38   Variation (σ/μ) 4.16e-01   Residual 1.93e-01
#      27   Volume ratio 12.43   Variation (σ/μ) 4.16e-01   Residual 1.93e-01
#      28   Volume ratio 12.44   Variation (σ/μ) 4.16e-01   Residual 1.93e-01
#      29   Volume ratio 12.44   Variation (σ/μ) 4.16e-01   Residual 1.93e-01
#      30   Volume ratio 12.44   Variation (σ/μ) 4.16e-01   Residual 1.93e-01
#      31   Volume ratio 12.44   Variation (σ/μ) 4.16e-01   Residual 1.93e-01
#      32   Volume ratio 12.44   Variation (σ/μ) 4.16e-01   Residual 1.93e-01
#      33   Volume ratio 12.44   Variation (σ/μ) 4.16e-01   Residual 1.93e-01
#      34   Volume ratio 12.44   Variation (σ/μ) 4.16e-01   Residual 1.93e-01
#      35   Volume ratio 13.34   Variation (σ/μ) 4.18e-01   Residual 1.94e-01
#      36   Volume ratio 15.21   Variation (σ/μ) 4.38e-01   Residual 2.09e-01
#      37   Volume ratio 17.20   Variation (σ/μ) 4.76e-01   Residual 2.22e-01
#      38   Volume ratio 11.89   Variation (σ/μ) 4.30e-01   Residual 1.45e-01
#      39   Volume ratio 12.41   Variation (σ/μ) 4.13e-01   Residual 1.01e-01
#      40   Volume ratio  8.76   Variation (σ/μ) 3.79e-01   Residual 7.36e-02
#      41   Volume ratio 10.00   Variation (σ/μ) 3.85e-01   Residual 6.36e-02
#      42   Volume ratio  8.57   Variation (σ/μ) 3.65e-01   Residual 5.17e-02
#      43   Volume ratio  9.48   Variation (σ/μ) 3.72e-01   Residual 4.17e-02
#      44   Volume ratio  8.70   Variation (σ/μ) 3.60e-01   Residual 3.51e-02
#      45   Volume ratio  9.31   Variation (σ/μ) 3.64e-01   Residual 2.69e-02
#      46   Volume ratio  8.82   Variation (σ/μ) 3.57e-01   Residual 2.23e-02
#      47   Volume ratio  9.20   Variation (σ/μ) 3.58e-01   Residual 1.44e-02
#      48   Volume ratio  8.95   Variation (σ/μ) 3.53e-01   Residual 1.03e-02
#      49   Volume ratio  9.08   Variation (σ/μ) 3.52e-01   Residual 4.91e-03
#      50   Volume ratio  9.05   Variation (σ/μ) 3.50e-01   Residual 3.98e-03
#      51   Volume ratio  9.13   Variation (σ/μ) 3.51e-01   Residual 2.81e-03
#      52   Volume ratio  9.11   Variation (σ/μ) 3.50e-01   Residual 2.28e-03
#      53   Volume ratio  9.17   Variation (σ/μ) 3.50e-01   Residual 1.51e-03
#      54   Volume ratio  9.15   Variation (σ/μ) 3.50e-01   Residual 1.11e-03
#      55   Volume ratio  9.19   Variation (σ/μ) 3.50e-01   Residual 5.28e-04
#      56   Volume ratio  9.18   Variation (σ/μ) 3.49e-01   Residual 2.89e-04
#      57   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 1.04e-04
#      58   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 7.69e-05
#      59   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 5.31e-05
#      60   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 3.80e-05
#      61   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 2.66e-05
#      62   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 1.87e-05
#      63   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 1.36e-05
#      64   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 9.55e-06
#      65   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 7.22e-06
#      66   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 5.04e-06
#      67   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 3.88e-06
#      68   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 2.50e-06
#      69   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 1.87e-06
#      70   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 9.04e-07
#      71   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 6.32e-07
#      72   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 2.67e-07
#      73   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 1.55e-07
#      74   Volume ratio  9.19   Variation (σ/μ) 3.49e-01   Residual 8.22e-08
#    Solver converged in 74 iterations.
#
# The adapted mesh can be accessed via the `mesh` attribute of the mover. Plotting it,
# we see that the adapted mesh has its resolution focused around a ring, as expected.

fig, axes = plt.subplots()
triplot(mover.mesh, axes=axes)
axes.set_aspect(1)
plt.savefig("monge_ampere_ring-adapted_mesh.jpg")

# .. figure:: monge_ampere_ring-adapted_mesh.jpg
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
plt.savefig("monge_ampere_ring-adapted_mesh_zoom.jpg")

# .. figure:: monge_ampere_ring-adapted_mesh_zoom.jpg
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
# In the `next demo <./monge_ampere_periodic.py.html>`__, we will demonstrate
# that the Monge-Ampère method can also be to periodic meshes.
#
# This tutorial can be dowloaded as a `Python script <monge_ampere_ring.py>`__.
#
#
# .. rubric:: References
#
# .. bibliography::
#    :filter: docname in docnames
