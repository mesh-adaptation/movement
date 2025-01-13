import unittest

import ufl
from firedrake.utility_meshes import UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh

from movement.tangling import MeshTanglingChecker


class TestTangling(unittest.TestCase):
    """
    Unit tests for mesh tangling checking.
    """

    def test_1_tangled_element_1d_error(self):
        checker = MeshTanglingChecker(UnitIntervalMesh(3), raise_error=True)
        checker.mesh.coordinates.dat.data[1] += 0.4  # Vertex 1: (1/3)
        with self.assertRaises(ValueError) as cm:
            checker.check()
        self.assertEqual(str(cm.exception), "Mesh has 1 tangled element.")

    def test_1_tangled_element_1d(self):
        checker = MeshTanglingChecker(UnitIntervalMesh(3), raise_error=False)
        self.assertEqual(checker.check(), 0)
        checker.mesh.coordinates.dat.data[1] += 0.4  # Vertex 1: (1/3)
        self.assertEqual(checker.check(), 1)

    def test_2_tangled_elements_1d(self):
        checker = MeshTanglingChecker(UnitIntervalMesh(3), raise_error=False)
        checker.mesh.coordinates.dat.data[1] += 0.4  # Vertex 1: (1/3)
        checker.mesh.coordinates.dat.data[3] -= 0.4  # Vertex 3: (1)
        self.assertEqual(checker.check(), 2)

    def test_flip_1d(self):
        checker = MeshTanglingChecker(UnitIntervalMesh(3), raise_error=False)
        (x,) = ufl.SpatialCoordinate(checker.mesh)
        checker.mesh.coordinates.interpolate(ufl.as_vector([1 - x]))
        self.assertEqual(checker.check(), checker.mesh.num_cells())

    def test_1_tangled_element_2d(self):
        checker = MeshTanglingChecker(UnitSquareMesh(3, 3), raise_error=False)
        self.assertEqual(checker.check(), 0)
        checker.mesh.coordinates.dat.data[3] += 0.2  # Vertex 3: (1/3, 1/3)
        self.assertEqual(checker.check(), 1)

    def test_2_tangled_elements_2d(self):
        checker = MeshTanglingChecker(UnitSquareMesh(3, 3), raise_error=False)
        checker.mesh.coordinates.dat.data[3][0] += 0.5  # Vertex 3: (1/3, 1/3)
        self.assertEqual(checker.check(), 2)

    def test_3_tangled_elements_2d(self):
        checker = MeshTanglingChecker(UnitSquareMesh(3, 3), raise_error=False)
        checker.mesh.coordinates.dat.data[3] += 0.5  # Vertex 3: (1/3, 1/3)
        self.assertEqual(checker.check(), 3)

    def test_flip_2d(self):
        checker = MeshTanglingChecker(UnitSquareMesh(3, 3), raise_error=False)
        x, y = ufl.SpatialCoordinate(checker.mesh)
        checker.mesh.coordinates.interpolate(ufl.as_vector([1 - x, y]))
        self.assertEqual(checker.check(), checker.mesh.num_cells())

    def test_double_flip_2d(self):
        checker = MeshTanglingChecker(UnitSquareMesh(3, 3), raise_error=False)
        x, y = ufl.SpatialCoordinate(checker.mesh)
        checker.mesh.coordinates.interpolate(ufl.as_vector([1 - x, 1 - y]))
        self.assertEqual(checker.check(), 0)

    def test_2_tangled_elements_3d(self):
        checker = MeshTanglingChecker(UnitCubeMesh(3, 3, 3), raise_error=False)
        self.assertEqual(checker.check(), 0)
        checker.mesh.coordinates.dat.data[14][2] -= 0.4  # Vertex 14: (0, 0, 1)
        self.assertEqual(checker.check(), 2)

    def test_6_tangled_elements_3d(self):
        checker = MeshTanglingChecker(UnitCubeMesh(3, 3, 3), raise_error=False)
        self.assertEqual(checker.check(), 0)
        checker.mesh.coordinates.dat.data[33][0] += (0.4,)  # Vertex 33: (1/3, 1/3, 1/3)
        self.assertEqual(checker.check(), 6)

    def test_flip_3d(self):
        checker = MeshTanglingChecker(UnitCubeMesh(3, 3, 3), raise_error=False)
        x, y, z = ufl.SpatialCoordinate(checker.mesh)
        checker.mesh.coordinates.interpolate(ufl.as_vector([1 - x, y, z]))
        self.assertEqual(checker.check(), checker.mesh.num_cells())

    def test_double_flip_3d(self):
        checker = MeshTanglingChecker(UnitCubeMesh(3, 3, 3), raise_error=False)
        x, y, z = ufl.SpatialCoordinate(checker.mesh)
        checker.mesh.coordinates.interpolate(ufl.as_vector([1 - x, 1 - y, z]))
        self.assertEqual(checker.check(), 0)

    def test_triple_flip_3d(self):
        checker = MeshTanglingChecker(UnitCubeMesh(3, 3, 3), raise_error=False)
        x, y, z = ufl.SpatialCoordinate(checker.mesh)
        checker.mesh.coordinates.interpolate(ufl.as_vector([1 - x, 1 - y, 1 - z]))
        self.assertEqual(checker.check(), checker.mesh.num_cells())
