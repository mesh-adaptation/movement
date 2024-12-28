import unittest

import ufl
from firedrake.utility_meshes import UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh

from movement.tangling import MeshTanglingChecker


class TestTangling(unittest.TestCase):
    """
    Unit tests for mesh tangling checking.
    """

    def test_dim_notimplementederror(self):
        with self.assertRaises(ValueError) as cm:
            MeshTanglingChecker(UnitCubeMesh(1, 1, 1))
        msg = "Mesh dimension 3 not supported."
        self.assertEqual(str(cm.exception), msg)

    def test_1_tangled_element_1d_error(self):
        checker = MeshTanglingChecker(UnitIntervalMesh(3), raise_error=True)
        checker.mesh.coordinates.dat.data[1] += 0.4
        with self.assertRaises(ValueError) as cm:
            checker.check()
        self.assertEqual(str(cm.exception), "Mesh has 1 tangled element.")

    def test_1_tangled_element_1d(self):
        checker = MeshTanglingChecker(UnitIntervalMesh(3), raise_error=False)
        self.assertEqual(checker.check(), 0)
        checker.mesh.coordinates.dat.data[1] += 0.4
        self.assertEqual(checker.check(), 1)

    def test_2_tangled_elements_1d(self):
        checker = MeshTanglingChecker(UnitIntervalMesh(3), raise_error=False)
        checker.mesh.coordinates.dat.data[1] += 0.4
        checker.mesh.coordinates.dat.data[3] -= 0.4
        self.assertEqual(checker.check(), 2)

    def test_1_tangled_element_2d(self):
        checker = MeshTanglingChecker(UnitSquareMesh(3, 3), raise_error=False)
        self.assertEqual(checker.check(), 0)
        checker.mesh.coordinates.dat.data[3] += 0.2
        self.assertEqual(checker.check(), 1)

    def test_2_tangled_elements_2d(self):
        checker = MeshTanglingChecker(UnitSquareMesh(3, 3), raise_error=False)
        checker.mesh.coordinates.dat.data[3][0] += 0.5
        self.assertEqual(checker.check(), 2)

    def test_3_tangled_elements_2d(self):
        checker = MeshTanglingChecker(UnitSquareMesh(3, 3), raise_error=False)
        checker.mesh.coordinates.dat.data[3] += 0.5
        self.assertEqual(checker.check(), 3)

    def test_flip(self):
        checker = MeshTanglingChecker(UnitSquareMesh(3, 3), raise_error=False)
        x, y = ufl.SpatialCoordinate(checker.mesh)
        checker.mesh.coordinates.interpolate(ufl.as_vector([1 - x, y]))
        self.assertEqual(checker.check(), checker.mesh.num_cells())
