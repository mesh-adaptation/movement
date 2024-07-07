# Monge-Ampere in an L-shaped domain
# ==================================

# NOTE: https://www.firedrakeproject.org/variational-problems.html#boundary-conditions-on-interior-facets

# TODO: text

from firedrake import *

from movement import *

# mesh, immersed_facets = Mesh("l.msh"), []
# mesh, immersed_facets = Mesh("l_convex.msh"), [3, 4]
mesh, immersed_facets = SquareMesh(20, 20, 2, 2), []
mesh.coordinates.dat.data[:] -= 1
VTKFile("monge_ampere_l-initial_mesh.pvd").write(mesh.coordinates)


def ring_monitor(mesh):
    alpha = Constant(20.0)
    beta = Constant(200.0)
    gamma = Constant(sqrt(0.15))
    x, y = SpatialCoordinate(mesh)
    r = x**2 + y**2
    return Constant(1.0) + alpha / cosh(beta * (r - gamma**2)) ** 2


P1 = FunctionSpace(mesh, "CG", 1)
m = Function(P1)
m.interpolate(ring_monitor(mesh))
VTKFile("monge_ampere_l-monitor.pvd").write(m)

# FIXME: Why are elements tangling?
rtol = 1.0e-03
mover = MongeAmpereMover(
    mesh,
    ring_monitor,
    method="quasi_newton",
    rtol=rtol,
    immersed_facet_tags=immersed_facets,
    raise_convergence_errors=False,
)
mover.move()
VTKFile("monge_ampere_l-adapted_mesh.pvd").write(mover.mesh.coordinates)
