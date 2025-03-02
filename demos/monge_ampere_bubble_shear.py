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
# mesh resolution aligned with the edge of the circle. ::

mb = RingMonitorBuilder(centre=(0.5, 0.65), radius=0.15, amplitude=10.0, width=200.0)
ring_monitor = mb.get_monitor()
rtol = 1.0e-03
mover = MongeAmpereMover(mesh, ring_monitor, method="quasi_newton", rtol=rtol)
mover.move()
mesh = mover.mesh

# TODO: Outputs

# Plotting this gives a similar result as in the first Monge-Ampere demo. ::

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
c = Function(Q).interpolate(initial_condition(mesh))
c_ = Function(Q).assign(c)
u = Function(V)
u_ = Function(V)

# The velocity field for the problem is an analyically prescribed rotational field, set
# up such that it is periodic in time, returning to its initial condition after one full
# period :math:`T`. Defining the current time as a variable in :math:`\mathbb{R}`-space,
# we can define the velocity expression in UFL once and then update the time as the
# solution progresses. ::

x, y = SpatialCoordinate(mesh)
T = Constant(6.0)  # period of the velocity field
R = FunctionSpace(mesh, "R", 0)
dt = Function(R).assign(0.01)  # timestep size
t = Function(R).assign(dt)  # current time
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
tau = 0.5 * h / U
tau = min_value(tau, U * h / (6 * D))
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

# To properly transfer the solution field when the mesh is moved, we need to stash a
# separate mesh and solution variable to hold the data. ::


class Stash:
    """
    Class for holding temporary mesh and solution data.
    """

    mesh = Mesh(mesh.coordinates.copy(deepcopy=True))
    solution = None


# For efficiency, define the Hessian, its norm, and the monitor scaling once, rather
# than inside the monitor function. ::

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

# Now we're ready to solve the PDE. ::

c_.interpolate(initial_condition(mesh))
c.assign(c_)
t_end = 6.0
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

# TODO: Discussion
#
# This tutorial can be dowloaded as a `Python script <monge_ampere_bubble_shear.py>`__.
#
#
# .. rubric:: References
#
# .. bibliography::
#    :filter: docname in docnames
