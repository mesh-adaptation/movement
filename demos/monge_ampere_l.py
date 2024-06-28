# Monge-Ampere in an L-shaped domain
# ==================================

# TODO: text

from firedrake import *

from movement import *

mesh = Mesh("l.msh")
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


rtol = 1.0e-08
mover = MongeAmpereMover(mesh, ring_monitor, method="quasi_newton", rtol=rtol)
mover.move()
VTKFile("monge_ampere_l-adapted_mesh.pvd").write(mover.mesh.coordinates)
