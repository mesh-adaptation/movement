# Monge-Ampere on a torus
# =======================

from firedrake import *

from movement import *

mesh = TorusMesh(100, 20, 2, 1)

VTKFile("torus_mesh.pvd").write(mesh.coordinates)

# TODO
