# Monge-Ampere in a domain with a hole
# ====================================

# TODO: text

from firedrake import *

from movement import *

mesh = Mesh("mesh-with-hole.msh")
p2coords = Function(VectorFunctionSpace(mesh, "CG", 2))
p2coords.interpolate(mesh.coordinates)
mesh = Mesh(p2coords)
# FIXME: Make P2 mesh properly - see Thetis example
VTKFile("mesh_with_hole.pvd").write(mesh.coordinates)


def ring_monitor(mesh):
    alpha = Constant(20.0)
    beta = Constant(2.0)
    gamma = Constant(0.25)
    x, y = SpatialCoordinate(mesh)
    r = (x - 2) ** 2 + (y - 0.5) ** 2
    return Constant(1.0) + alpha / cosh(beta * (r - gamma**2)) ** 2


P1 = FunctionSpace(mesh, "CG", 1)
m = Function(P1)
m.interpolate(ring_monitor(mesh))
VTKFile("monitor.pvd").write(m)


rtol = 1.0e-08
mover = MongeAmpereMover(mesh, ring_monitor, method="quasi_newton", rtol=rtol)
mover.move()
VTKFile("adapted_mesh.pvd").write(mover.mesh.coordinates)
