# Using mesh movement to optimize the mesh for PDE solution
# =========================================================
#
# In this demo will demonstrate how we might use mesh movement
# to obtain a mesh that is optimized for solving a particular
# PDE. The general idea is that we want to reduce numerical
# error by increasing resolution where the local error is
# (expected to be) high and decrease it elsewhere.
#
# As an example we will look at the discretisation of the
# Helmholtz equation
#
# .. math::
#
#    -\nabla^2 u + u &= f
#
#    \nabla u \cdot \vec{n} &= 0 \quad \textrm{on}\ \Gamma
#
# For an explanation of how we can use Firedrake to implement a Finite Element
# Method (FEM) discretisation of this PDE see the `corresponding Firedrake demo
# <https://www.firedrakeproject.org/demos/helmholtz.py.html>`_. The only changes
# we introduce is that we choose a different, slightly more interesting
# solution :math:`u` and rhs :math:`f`
#
# .. math::
#
#    u(x, y) &= \exp(-((x-0.5)^2 + (y-0.5)^2))
#
#    f(x, y) &= \left[ -\frac{4(x-0.5)^2 + 4(y-0.5)^2}{w^4} + \frac{4}{w^2} + 1 \right]
#               \exp(-((x-0.5)^2 + (y-0.5)^2))
#
# Note that here we *first* choose the solution :math:`u` after which we can compute what
# rhs :math:`f` should be, by substitution in the PDE, in order for :math:`u` to be the
# analytical solution. This so called Method of Manufactured Solutions approach is an
# easy way to construct PDE configurations for which we know the analytical solution,
# e.g. for testing purposes.
#
# Based on the code in the Firedrake demo, we first solve the PDE on a uniform mesh.
# Because our chosen solution does not satisfy homogenoues Neumann boundary conditions,
# we instead apply Dirichlet boundary conditions based on the chosen analytical solution.

from firedrake import *
from movement import MongeAmpereMover

mesh = UnitSquareMesh(20, 20)  # initial mesh


def u_exact(mesh):
    """Return UFL expression of exact solution"""
    x, y = SpatialCoordinate(mesh)
    # some arbitrarily chosen centre (x0, y0) and width w
    w = Constant(0.1)
    x0 = Constant(0.51)
    y0 = Constant(0.65)
    return exp(-((x - x0) ** 2 + (y - y0) ** 2) / w**2)


def solve_helmholtz(mesh):
    """Solve the Helmholtz PDE on the given mesh"""
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    u_exact_expr = u_exact(mesh)
    f = -div(grad(u_exact_expr)) + u_exact_expr
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = inner(f, v) * dx
    u = Function(V)
    bcs = DirichletBC(V, u_exact_expr, "on_boundary")
    solve(a == L, u, bcs=bcs)
    return u


u_h = solve_helmholtz(mesh)

import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor

fig, axes = plt.subplots()
contours = tripcolor(u_h, axes=axes)
fig.colorbar(contours)
plt.savefig("monge_ampere_helmholtz-initial_solution.jpg")

# .. figure:: monge_ampere_helmholtz-initial_solution.jpg
#    :figwidth: 60%
#    :align: center
#
# As in the Helmholtz demo, we can compute the L2-norm of the numerical error:

error = u_h - u_exact(mesh)
print("L2-norm error on initial mesh:", sqrt(assemble(dot(error, error) * dx)))

# We will now try to use mesh movement to optimize the mesh to reduce
# this numerical error. A good indicator for where resolution is required
# is to look at the curvature of the solution which can be expressed
# in terms of the norm of the Hessian. A monitor function
# that targets high resolution in places of high curvature then looks like
#
# .. math::
#
#    m = 1 + \alpha \frac{H(u_h):H(u_h)}{\max_{{\bf x}\in\Omega} H(u):H(u)}
#
# where :math:`:` indicates the inner product, i.e. :math:`\sqrt{H:H}` is the Frobenius norm
# of :math:`H`. We have normalized such that the minimum of the monitor function is one (where
# the error is zero), and its maximum is :math:`1 + \alpha` (where the curvature is maximal). This
# means we can select the ratio between the largest area and smallest area triangle in the
# moved mesh as :math:`1+\alpha`.
#
# In the following implementation we use the exact solution :math:`u_{\text{exact}}` which we
# have as a symbolic UFL expression, and thus we can also obtain the Hessian symbolically as
# :code:`grad(grad(u_exact))`. To compute its maximum norm however we do interpolate it
# to a P1 function space `V` and take the maximum of the array of nodal values.

alpha = Constant(5.0)


def monitor(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    Hnorm = Function(V, name="Hnorm")
    H = grad(grad(u_exact(mesh)))
    Hnorm.interpolate(sqrt(inner(H, H)))
    Hnorm_max = Hnorm.dat.data.max()
    m = 1 + alpha * Hnorm / Hnorm_max
    return m


fig, axes = plt.subplots()
m = Function(u_h, name="monitor")
m.interpolate(monitor(mesh))
contours = tripcolor(m, axes=axes)
fig.colorbar(contours)
plt.savefig("monge_ampere_helmholtz-monitor.jpg")
fig, axes = plt.subplots()

# .. figure:: monge_ampere_helmholtz-monitor.jpg
#    :figwidth: 60%
#    :align: center
#
# As in the `previous Monge-Ampère demo <./monge_ampere1.py.html>`__, we use the
# MongeAmpereMover to perform the mesh movement based on this monitor. We need
# to provide the monitor as a callback function that takes the mesh as its
# input just as we have defined it above. During the iterations of the mesh
# movement process the monitor will then be re-evaluated in the (iteratively)
# moved mesh nodes so that, as we improve the mesh, we can also more accurately
# express the monitor function in the desired high-resolution areas.

mover = MongeAmpereMover(mesh, monitor, method="quasi_newton")
mover.move()

# Plotting the resulting mesh

from firedrake.pyplot import triplot

fig, axes = plt.subplots()
triplot(mover.mesh, axes=axes)
axes.set_aspect(1)
plt.savefig("monge_ampere_helmholtz-adapted_mesh.jpg")

# .. figure:: monge_ampere_helmholtz-adapted_mesh.jpg
#    :figwidth: 60%
#    :align: center
#
# Now let us see whether the numerical error has actually been reduced
# if we solve the PDE on this mesh:

u_h = solve_helmholtz(mover.mesh)
error = u_h - u_exact(mover.mesh)
print("L2-norm error on moved mesh:", sqrt(assemble(dot(error, error) * dx)))

# Of course, in many practical problems we do not actually have access
# to the exact solution. We can then use the Hessian of the numerical
# solution in the monitor function. Calculating the Hessian we have to
# be a bit careful however: since our numerical FEM solution :math:`u_h`
# is a piecewise linear function, its first gradient results in a
# piecewise *constant* function, a vector-valued constant
# function in each cell. Taking its gradient in each cell would therefore
# simply be zero. Instead, we should numerically approximate the derivatives
# to "recover" the Hessian, for which there are a number of different methods.
#
# As Hessians are often used in metric-based h-adaptivity, some of these
# methods have been implemented in the :py:mod:`animate` package.
# An implementation of a monitor based on the Hessian of the numerical
# solution is given below. Note that this requires solving the Helmholtz
# PDE in every mesh movement iteration, and thus may become significantly
# slower for large problems.

from animate import RiemannianMetric


def monitor2(mesh):
    u_h = solve_helmholtz(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    TV = TensorFunctionSpace(mesh, "CG", 1)
    H = RiemannianMetric(TV)
    H.compute_hessian(u_h, method="L2")

    Hnorm = Function(V, name="Hnorm")
    Hnorm.interpolate(sqrt(inner(H, H)))
    Hnorm_max = Hnorm.dat.data.max()
    m = 1 + alpha * Hnorm / Hnorm_max
    return m


mover = MongeAmpereMover(mesh, monitor2, method="quasi_newton")
mover.move()

u_h = solve_helmholtz(mover.mesh)
error = u_h - u_exact(mover.mesh)
print("L2-norm error on moved mesh:", sqrt(assemble(dot(error, error) * dx)))
