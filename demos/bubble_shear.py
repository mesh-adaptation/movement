from animate.metric import RiemannianMetric
from firedrake import *

from movement import MongeAmpereMover


def monitor(mesh):
    alpha = Constant(50.0)

    V = FunctionSpace(mesh, "CG", 1)
    Hnorm = Function(V, name="Hnorm")

    # Define a dummy metric if H has not been defined yet (before the first timestep)
    if "H" not in globals():
        _H = RiemannianMetric(TensorFunctionSpace(mesh, "CG", 1))
        _H.interpolate(Constant(((1.0, 0.0), (0.0, 1.0))))
    else:
        _H = H

    Hnorm.interpolate(sqrt(inner(_H, _H)))
    Hnorm_max = Hnorm.dat.data.max()
    m = 1 + alpha * Hnorm / Hnorm_max
    return m


mesh = UnitSquareMesh(64, 64)
mover = MongeAmpereMover(mesh, monitor, method="quasi_newton", rtol=1e-8)

Q = FunctionSpace(mover.mesh, "CG", 1)
V = VectorFunctionSpace(mover.mesh, "CG", 1)
R = FunctionSpace(mover.mesh, "R", 0)

c = Function(Q)  # solution at current timestep
c_ = Function(Q)  # previous timestep

# Initial condition
x, y = SpatialCoordinate(mover.mesh)
ball_r, ball_x0, ball_y0 = 0.15, 0.5, 0.65
r = sqrt(pow(x - ball_x0, 2) + pow(y - ball_y0, 2))
ic = conditional(r < ball_r, 1.0, 0.0)
c_.interpolate(ic)

u = Function(V)  # velocity field at current timestep
u_ = Function(V)  # velocity field at previous timestep

T = Constant(6.0)  # period of the velocity field
dt = Function(R).assign(0.01)  # timestep size
t = Function(R).assign(dt)  # current time
t_end = 0.15  # simulation end time
theta = Function(R).assign(0.5)  # Crank-Nicolson implicitness

# Expressions for the velocity field
u_expression = as_vector(
    [
        2 * sin(pi * x) ** 2 * sin(2 * pi * y) * cos(2 * pi * t / T),
        -sin(2 * pi * x) * sin(pi * y) ** 2 * cos(2 * pi * t / T),
    ]
)

# SUPG stabilisation
D = Function(R).assign(0.1)  # diffusivity coefficient
h = CellSize(mover.mesh)  # mesh cell size
U = sqrt(dot(u, u))  # velocity magnitude
tau = 0.5 * h / U
tau = min_value(tau, U * h / (6 * D))

# Apply SUPG stabilisation to the test function
phi = TestFunction(Q)
phi += tau * dot(u, grad(phi))

# Variational form of the advection equation
trial = TrialFunction(Q)
a = inner(trial, phi) * dx + dt * theta * inner(dot(u, grad(trial)), phi) * dx
L = inner(c_, phi) * dx - dt * (1 - theta) * inner(dot(u_, grad(c_)), phi) * dx

lvp = LinearVariationalProblem(a, L, c, bcs=DirichletBC(Q, 0.0, "on_boundary"))
lvs = LinearVariationalSolver(lvp)

# Initialise the metric
H = RiemannianMetric(TensorFunctionSpace(mover.mesh, "CG", 1))

outfile = VTKFile("bubble_shear-moved.pvd")

while float(t) < t_end + 0.5 * float(dt):
    # Update the background velocity field at the current timestep
    u.interpolate(u_expression)

    # Solve the advection equation
    lvs.solve()

    H.compute_hessian(c)
    mover.move()

    # Export the solution
    outfile.write(c, time=float(t))

    # Update the solution at the previous timestep
    c_.assign(c)
    u_.assign(u)
    t.assign(t + dt)
