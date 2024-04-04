from firedrake import *
from movement import *


def test_tangling_checker():
    """
    Test that :class:`MeshTanglingChecker`
    can correctly identify tangled elements
    in a 2D triangular mesh.
    """
    mesh = UnitSquareMesh(3, 3)
    checker = MeshTanglingChecker(mesh, raise_error=False)
    assert checker.check() == 0
    mesh.coordinates.dat.data[3] += 0.2
    assert checker.check() == 1
