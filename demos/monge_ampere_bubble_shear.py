# Mesh Movement in a time-dependent problem
# =========================================
#
# In the `previous demo <./monge_ampere_helmholtz.py.html>`__, we demonstrated mesh
# movement with the Monge-Ampère method in the context of a steady-state PDE. In this
# demo, we progress to the time-dependent case.
#
# The demo solves an advection equation,
#
# ..math::
#    \frac{\partial c}{\partial t}+\mathbf{u}\cdot\nabla c=0,
#
# where the solution field :math:`c` is approximated in :math:`\mathbb{P}1` space, and
# the background velocity field is denoted by :math:`\mathbf{u}`. The velocity field is
# approximated in vector :math:`\mathbb{P}1` space.
#
# As in previous examples, import from Firedrake and the :class:`~.MongeAmpereMover`. In
# this demo, we also import :class:`~.RiemannianMetric` from the Animate metric-based
# mesh adaptation package, as it comes with the functionality to recover Hessians. ::

from animate.metric import RiemannianMetric
from firedrake import *

from movement.monge_ampere import MongeAmpereMover
from movement.monitor import RingMonitorBuilder

# As in most demos, we define a uniform mesh of the unit square. ::

n = 20
mesh = UnitSquareMesh(n, n)

# The initial condition for the problem is unity in a circle shape of radius 0.15,
# centred at :math:`(0.5,0.65)` and zero elsewhere. ::

ball_r, ball_x0, ball_y0 = 0.15, 0.5, 0.65


def initial_condition(mesh2d):
    x, y = SpatialCoordinate(mesh2d)
    r = sqrt(pow(x - ball_x0, 2) + pow(y - ball_y0, 2))
    return conditional(r < ball_r, 1.0, 0.0)


# Given this initial condition, it makes sense to move the initial mesh so that it has
# mesh resolution aligned with the edge of the circle. (Recall the earlier
# `ring demo <./monge_ampere_ring.py.html>`__.) We keep the relative tolerance used in
# the Monge-Ampère solver to :math:`10^{-3}` for computational efficiency. ::

mb = RingMonitorBuilder(centre=(0.5, 0.65), radius=0.15, amplitude=10.0, width=200.0)
ring_monitor = mb.get_monitor()
rtol = 1.0e-03
mover = MongeAmpereMover(mesh, ring_monitor, method="quasi_newton", rtol=rtol)
mover.move()
mesh = mover.mesh

# .. code-block:: none
#
#       0   Volume ratio  7.19   Variation (σ/μ) 8.77e-01   Residual 8.55e-01
#       1   Volume ratio  7.60   Variation (σ/μ) 5.80e-01   Residual 5.39e-01
#       2   Volume ratio  7.39   Variation (σ/μ) 5.27e-01   Residual 4.54e-01
#       3   Volume ratio  4.67   Variation (σ/μ) 4.31e-01   Residual 3.49e-01
#       4   Volume ratio  4.96   Variation (σ/μ) 3.95e-01   Residual 2.90e-01
#       5   Volume ratio  4.44   Variation (σ/μ) 3.36e-01   Residual 2.25e-01
#       6   Volume ratio  4.03   Variation (σ/μ) 2.98e-01   Residual 1.70e-01
#       7   Volume ratio  3.51   Variation (σ/μ) 2.56e-01   Residual 1.14e-01
#       8   Volume ratio  3.52   Variation (σ/μ) 2.24e-01   Residual 7.84e-02
#       9   Volume ratio  3.43   Variation (σ/μ) 1.98e-01   Residual 4.80e-02
#      10   Volume ratio  3.27   Variation (σ/μ) 1.83e-01   Residual 2.91e-02
#      11   Volume ratio  3.11   Variation (σ/μ) 1.72e-01   Residual 1.50e-02
#
# Plotting the adapted mesh gives a similar result as in the `earlier demo
# <./monge_ampere_ring.py.html>`__. ::

import matplotlib.pyplot as plt

fig, axes = plt.subplots()
triplot(mesh, axes=axes)
axes.set_aspect(1)
plt.savefig("monge_ampere_bubble_shear-initial_adapted_mesh.jpg")

# .. figure:: monge_ampere_bubble_shear-initial_adapted_mesh.jpg
#    :figwidth: 60%
#    :align: center
#
# With a suitable initial mesh, we can now define the PDE problem.
#
# We denote the solution at the current timestep by `c` and the lagged solution (at the
# previous timestep) by `c_`. Similarly, we use `u` to denote the velocity field at the
# current timestep and `u_` to denote the lagged velocity field. ::

Q = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 1)
c = Function(Q, name="concentration").interpolate(initial_condition(mesh))
c_ = Function(Q).assign(c)
u = Function(V, name="velocity")
u_ = Function(V)

# The velocity field for the problem is an analyically prescribed rotational field, set
# up such that it is periodic in time, returning to its initial condition after half of
# the full period :math:`T`. Defining the current time as a variable in
# :math:`\mathbb{R}`-space, we can define the velocity expression in UFL once and then
# update the time as the solution progresses. ::

x, y = SpatialCoordinate(mesh)
T = Constant(6.0, name="period")
R = FunctionSpace(mesh, "R", 0)
dt = Function(R, name="timestep").assign(0.01)
t = Function(R, name="current time").assign(dt)
u_expression = as_vector(
    [
        2 * sin(pi * x) ** 2 * sin(2 * pi * y) * cos(2 * pi * t / T),
        -sin(2 * pi * x) * sin(pi * y) ** 2 * cos(2 * pi * t / T),
    ]
)

# To stabilise the :math:`\mathbb{P}1` discretisation, we make use of SUPG with an
# artificial diffusivity :math:`D = 0.1`. ::

D = Function(R).assign(0.1)
h = CellSize(mesh)
U = sqrt(dot(u, u))
tau = min_value(0.5 * h / U, U * h / (6 * D))
phi = TestFunction(Q)
phi += tau * dot(u, grad(phi))

# We are now able to define the vriational form of the advection equation. ::

trial = TrialFunction(Q)
theta = Function(R).assign(0.5)  # Crank-Nicolson implicitness
a = inner(trial, phi) * dx + dt * theta * inner(dot(u, grad(trial)), phi) * dx
L = inner(c_, phi) * dx - dt * (1 - theta) * inner(dot(u_, grad(c_)), phi) * dx
lvp = LinearVariationalProblem(a, L, c, bcs=DirichletBC(Q, 0.0, "on_boundary"))

# For the linear solver, it should be sufficient to use GMRES preconditioned with SOR.
# ::

solver_parameters = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "pc_type": "sor",
}
lvs = LinearVariationalSolver(lvp, solver_parameters=solver_parameters)

# To properly transfer the solution field when the mesh is moved within the time
# integration loop, we need to stash a separate mesh and solution variable to hold the
# data. ::


class Stash:
    """
    Class for holding temporary mesh and solution data.
    """

    mesh = Mesh(mesh.coordinates.copy(deepcopy=True))
    solution = None


# For the subsequent mesh movement, we switch to a similar Hessian-based monitor
# function as considered in the `previous demo
# <./monge_ampere_helmholtz.py.html>`__. For efficiency, define the Hessian, its norm,
# and the monitor scaling once, rather than inside the monitor function. ::

P1_ten = TensorFunctionSpace(mesh, "CG", 1)
H = RiemannianMetric(P1_ten)
Hnorm = Function(Q, name="Hnorm")
alpha = Constant(10.0)


def monitor(mesh):
    # Project the stashed solution data onto the current mesh
    Q_tmp = FunctionSpace(Stash.mesh, "CG", 1)
    Stash.solution = Function(Q_tmp)
    Stash.solution.dat.data[:] = c.dat.data
    c.project(Stash.solution)

    # Compute the monitor expression using the Hessian
    H.compute_hessian(c)
    Hnorm.interpolate(sqrt(inner(H, H)))
    Hnorm_max = Hnorm.dat.data.max()
    m = 1 + alpha * Hnorm / Hnorm_max

    # Stash the mesh before it's adapted
    Stash.mesh = Mesh(mesh.coordinates.copy(deepcopy=True))
    return m


# Switch the monitor function used by the :class:`~.MongeAmpereMover`. ::

mover.monitor_function = monitor

# Now we're ready to solve the PDE. Let's just solve for a quarter period, which is
# sufficient to see the bubble of tracer concentration stretch out considerably. ::

t_end = 1.5
c_.interpolate(initial_condition(mesh))
c.assign(c_)
outfile = VTKFile("bubble_shear.pvd")
outfile.write(c)
while float(t) < t_end + 0.5 * float(dt):
    # Update the background velocity field at the current timestep
    u.interpolate(u_expression)

    # Solve the advection equation
    lvs.solve()

    # Move the mesh
    mover.move()

    # Export the solution
    outfile.write(c, time=float(t))

    # Update the solution at the previous timestep
    c_.assign(c)
    u_.assign(u)
    t.assign(t + dt)

# The outputs of the mesh movement steps are not shown here, for brevity. However,
# looking at a small sample, we observe that very few iterations are required once the
# mesh movement has become established:
#
# .. code-block:: none
#
#    ...
#       0   Volume ratio  2.24   Variation (σ/μ) 7.24e-02   Residual 3.08e-02
#       1   Volume ratio  2.09   Variation (σ/μ) 7.02e-02   Residual 1.32e-02
#    Solver converged in 1 iteration.
#       0   Volume ratio  2.23   Variation (σ/μ) 6.73e-02   Residual 2.95e-02
#       1   Volume ratio  1.98   Variation (σ/μ) 6.35e-02   Residual 1.24e-02
#    Solver converged in 1 iteration.
#    ...


fig, axes = plt.subplots()
tricontourf(c, axes=axes)
axes.set_aspect(1)
plt.savefig("monge_ampere_bubble_shear-final_solution.jpg")

# .. figure:: monge_ampere_bubble_shear-final_solution.jpg
#    :figwidth: 60%
#    :align: center

fig, axes = plt.subplots()
triplot(mesh, axes=axes)
axes.set_aspect(1)
plt.savefig("monge_ampere_bubble_shear-final_adapted_mesh.jpg")

# .. figure:: monge_ampere_bubble_shear-final_adapted_mesh.jpg
#    :figwidth: 60%
#    :align: center
#
# As you might have already realised, the approach documented here is a completely
# over-the-top way to solve a simple linear advection problem. However, it shows how the
# Monge-Ampère approach might be used in other problems for which it brings a greater
# advantage.
#
# Given that the PDE represents linear advection, this example is a prime candidate for
# Lagrangian-type mesh movement approaches, which are likely to have far lower
# computational costs than a full Monge-Ampère approach as used here.
#
# This tutorial can be dowloaded as a `Python script <monge_ampere_bubble_shear.py>`__.
#
#
# .. rubric:: References
#
# .. bibliography::
#    :filter: docname in docnames
