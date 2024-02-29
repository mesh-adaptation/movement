# Mesh movement using Laplacian smoothing
# =======================================
#
# In this demo, we again consider the simple boundary forcing problem from the
# `lineal spring demo <lineal_spring.py.html>`__. However, instead of using the lineal
# spring method, we use Laplacian smoothing.
#
# Recall that the lineal spring approach interprets the mesh as a fictitious structure
# of beams, assigning the beams lengths and stiffness values. Given some forcing of the
# beams, we determine how the rest of the structure responds by solving a discrete
# Poisson equation. The difference with the Laplacian smoothing approach is that we do
# not make use of the beam structure and simply formulate the Poisson equation in terms
# of updates to the mesh coordinates, via a *mesh velocity*. That is, we solve
#
# .. math::
#    -\Delta\mathbf{v} = f
#
# and then compute
#
# .. math::
#    \mathbf{u} := \mathbf{u} + \mathbf{v}
#
# where :math:`\mathbf{u}` is the mesh coordinate field, :math:`\mathbf{v}` is the mesh
# velocity that we solve for under the forcing :math:`f`.
#
# As ever, we begin by importing from the namespaces of Firedrake and Movement. We also
# import various plotting utilites. ::

from firedrake import *
from movement import *
from firedrake.pyplot import triplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Recall the initial uniform mesh from the lineal spring demo, which is used again here.
# ::

mesh = UnitSquareMesh(10, 10)
coord_data_init = mesh.coordinates.dat.data.copy()

# .. figure:: lineal_spring-initial_mesh.jpg
#    :figwidth: 50%
#    :align: center
#
# In the lineal spring demo, the forcing corresponds to vertical
# movement of the top boundary, with the displacement following a sinusoidal pattern
# along that boundary. The forcing is also sinusoidal in time, such that it ramps up
# and then reverses, with the analytical solution being that the final mesh coincides
# with the initial mesh.

import numpy as np

A = 50  # forcing amplitude
T = 1.0  # forcing period


def forcing(x, t):
    return A * np.sin(2 * pi * t / T) * np.sin(pi * x)


num_timesteps = 10
dt = T / num_timesteps
times = np.arange(0, T + 0.5 * dt, dt)

# .. figure:: lineal_spring-forcings.jpg
#    :figwidth: 60%
#    :align: center
#
# The setup for the :class:`~.LaplacianSmoother` class is very similar to that for the
# :class:`~.SpringMover class. The only difference is that we need to specify the
# timestep value in addition to the mesh. Since we are going to apply a forcing to the
# top boundary, we need to extract the indices for the associated boundary nodes. We
# can then define the :func:`update_forcings` as in the other demo. ::

mover = LaplacianSmoother(mesh, dt)
boundary_nodes = DirichletBC(mesh.coordinates.function_space(), 0, 4).nodes


def update_forcings(t):
    coord_data = mesh.coordinates.dat.data
    forcing_data = mover.f.dat.data
    for i in boundary_nodes:
        x, y = coord_data[i]
        forcing_data[i][1] = forcing(x, t)


# We are now able to apply the mesh movement method. This works just as before. ::
# TODO: displacement

time = 0.0
fig, axes = plt.subplots(ncols=4, nrows=3, figsize=(12, 10))
for i, t in enumerate(times):
    idx = 0 if i == 0 else (i + 1)

    # Move the mesh and calculate the mesh speed
    mover.move(t, update_forcings=update_forcings, fixed_boundaries=[1, 2, 3])

    # Plot the current mesh, adding a time label
    ax = axes[idx // 4, idx % 4]
    triplot(mover.mesh, axes=ax)
    ax.legend(handles=[mpatches.Patch(label=f"t={t:.1f}")], handlelength=0)
    ax.set_ylim([-0.05, 1.45])
axes[0, 1].axis(False)
plt.savefig("laplacian_smoothing-adapted_meshes.jpg")

# .. figure:: laplacian_smoothing-adapted_meshes.jpg
#    :figwidth: 100%
#    :align: center
#
# Again, the mesh is deformed according to the vertical forcing, with the left, right,
# and bottom boundaries remaining fixed, returning it to be very close to its original
# state after one period. Let's check this in the :math:`\ell_\infty` norm. ::

coord_data = mover.mesh.coordinates.dat.data
linf_error = np.max(np.abs(coord_data - coord_data_init))
print(f"l_infinity error: {linf_error:.3f} m")
assert linf_error < 0.01

# This tutorial can be downloaded as a `Python script <laplacian_smoothing.py>`__.
