# Laplacian smoothing
# ===================
#
# Blah

from firedrake import *
from movement import *


num_timesteps = 10
dt = 1.0 / num_timesteps

mesh = UnitSquareMesh(10, 10)

A = 0.2  # forcing amplitude
T = 1.0  # forcing period


def forcing(x, t):
    return A * np.sin(2 * pi * t / T) * np.sin(pi * x)


times = np.arange(0, 1.0 + 0.5 * dt, dt)

mover = LaplacianSmoother(mesh, timestep=dt)
boundary_nodes = DirichletBC(mesh.coordinates.function_space(), 0, 4).nodes


def update_forcings(t):
    coord_data = mesh.coordinates.dat.data
    forcing_data = mover.f.dat.data
    for i in boundary_nodes:
        x, y = coord_data[i]
        forcing_data[i][1] = forcing(x, t)


from firedrake.pyplot import triplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

time = 0.0
fig, axes = plt.subplots(ncols=4, nrows=3, figsize=(12, 10))
for i, t in enumerate(times):
    idx = 0 if i == 0 else (i + 1)

    # Move the mesh and calculate the mesh speed
    mover.move(t, update_forcings=update_forcings, fixed_boundaries=[1, 2, 3])
    v_x = mover.v.dat.data[:, 0]
    v_y = mover.v.dat.data[:, 1]
    spd = np.sqrt(np.dot(v_x, v_x) + np.dot(v_y, v_y))
    print(f"time = {t:.1f} s, mesh speed = {spd:.1f}")

    # Plot the current mesh, adding a time label
    ax = axes[idx // 4, idx % 4]
    triplot(mover.mesh, axes=ax)
    ax.legend(handles=[mpatches.Patch(label=f"t={t:.1f}")], handlelength=0)
    ax.set_ylim([-0.05, 1.45])
axes[0, 1].axis(False)
plt.show()
plt.savefig("laplacian_smoothing-adapted_meshes.jpg")

# .. figure:: laplacian_smoothing-adapted_meshes.jpg
#    :figwidth: 100%
#    :align: center
#
