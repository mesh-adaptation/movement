from animate.metric import RiemannianMetric
from firedrake import *

from movement import MongeAmpereMover

T = 6.0


def velocity_expression(mesh, t):
    x, y = SpatialCoordinate(mesh)
    u_expr = as_vector(
        [
            2 * sin(pi * x) ** 2 * sin(2 * pi * y) * cos(2 * pi * t / T),
            -sin(2 * pi * x) * sin(pi * y) ** 2 * cos(2 * pi * t / T),
        ]
    )
    return u_expr


def get_initial_condition(mesh):
    x, y = SpatialCoordinate(mesh)
    ball_r, ball_x0, ball_y0 = 0.15, 0.5, 0.65
    r = sqrt(pow(x - ball_x0, 2) + pow(y - ball_y0, 2))
    c0 = Function(FunctionSpace(mesh, "CG", 1))
    c0.interpolate(conditional(r < ball_r, 1.0, 0.0))
    return c0


def run_simulation(mesh, t_start, t_end, c0):
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)

    c_ = Function(Q).project(c0)  # project initial condition onto the current mesh
    c = Function(Q)  # solution at current timestep

    t = Function(R).assign(t_start)
    u_expression = velocity_expression(mesh, t)  # velocity at t_start
    u_ = Function(V).interpolate(u_expression)
    u = Function(V)  # velocity field at current timestep

    # SUPG stabilisation
    D = Function(R).assign(0.1)  # diffusivity coefficient
    h = CellSize(mesh)  # mesh cell size
    U = sqrt(dot(u, u))  # velocity magnitude
    tau = 0.5 * h / U
    tau = min_value(tau, U * h / (6 * D))

    # Apply SUPG stabilisation to the test function
    phi = TestFunction(Q)
    phi += tau * dot(u, grad(phi))

    # Time-stepping parameters
    dt = Function(R).assign(0.01)  # timestep size
    theta = Function(R).assign(0.5)  # Crank-Nicolson implicitness

    # Variational form of the advection equation
    trial = TrialFunction(Q)
    a = inner(trial, phi) * dx + dt * theta * inner(dot(u, grad(trial)), phi) * dx
    L = inner(c_, phi) * dx - dt * (1 - theta) * inner(dot(u_, grad(c_)), phi) * dx

    # Define variational problem
    lvp = LinearVariationalProblem(a, L, c, bcs=DirichletBC(Q, 0.0, "on_boundary"))
    lvs = LinearVariationalSolver(lvp)

    # Integrate from t_start to t_end
    t.assign(t + dt)
    while float(t) < t_end + 0.5 * float(dt):
        # Update the background velocity field at the current timestep
        u.interpolate(u_expression)

        # Solve the advection equation
        lvs.solve()

        yield c

        # Update the solution at the previous timestep
        c_.assign(c)
        u_.assign(u)
        t.assign(t + dt)

    return c


def monitor(mesh):
    # print('monitor mesh:', mesh)
    alpha = Constant(50.0)

    # _c = Function(FunctionSpace(mesh, "CG", 1)).interpolate(c_mov)

    # P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    # H = RiemannianMetric(P1_ten)
    # try:
    #     H.compute_hessian(c_mov)
    # except NameError:
    #     H.compute_hessian(get_initial_condition(mesh))
    H_interp = RiemannianMetric(TensorFunctionSpace(mesh, "CG", 1))
    try:
        H_interp.interpolate(H)
    except NameError:
        H_interp.interpolate(Constant(((1.0, 0.0), (0.0, 1.0))))

    V = FunctionSpace(mesh, "CG", 1)
    Hnorm = Function(V, name="Hnorm")
    # Hnorm.interpolate(sqrt(inner(H, H)))
    Hnorm.interpolate(sqrt(inner(H_interp, H_interp)))
    Hnorm_max = Hnorm.dat.data.max()
    m = 1 + alpha * Hnorm / Hnorm_max
    return m


simulation_end_time = T / 20.0  # 2.0
mesh = UnitSquareMesh(64, 64)
mover = MongeAmpereMover(mesh, monitor, method="quasi_newton", rtol=1e-8)
c_mov = get_initial_condition(mover.mesh)
movement_simulation = run_simulation(mover.mesh, 0.0, simulation_end_time, c_mov)
t = 0
H = RiemannianMetric(TensorFunctionSpace(mover.mesh, "CG", 1))
outfile = VTKFile("bubble_shear-moved.pvd")
while True:
    try:
        print(t)
        c_mov = next(movement_simulation)
        H.compute_hessian(c_mov)
        mover.move()
        t += 0.01
        outfile.write(c_mov, time=t)
    except StopIteration:
        break
