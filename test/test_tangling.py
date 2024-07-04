import unittest

from firedrake import *

from movement import *


class TestTangling(unittest.TestCase):
    """
    Unit tests for mesh tangling checking.
    """

    def test_tangling_checker_error1(self):
        checker = MeshTanglingChecker(UnitSquareMesh(3, 3), raise_error=True)
        checker.mesh.coordinates.dat.data[3] += 0.2
        with self.assertRaises(ValueError) as cm:
            checker.check()
        self.assertEqual(str(cm.exception), "Mesh has 1 tangled element.")

    def test_tangling_checker_error2(self):
        checker = MeshTanglingChecker(UnitSquareMesh(3, 3), raise_error=True)
        checker.mesh.coordinates.dat.data[3] += 0.5
        with self.assertRaises(ValueError) as cm:
            checker.check()
        self.assertEqual(str(cm.exception), "Mesh has 3 tangled elements.")

    def test_tangling_checker_dim_error(self):
        with self.assertRaises(NotImplementedError) as cm:
            MeshTanglingChecker(UnitIntervalMesh(1))
        msg = "Tangling check only currently supported in 2D."
        self.assertEqual(str(cm.exception), msg)

    def test_tangling_checker_warning1(self):
        checker = MeshTanglingChecker(UnitSquareMesh(3, 3), raise_error=False)
        self.assertEqual(checker.check(), 0)
        checker.mesh.coordinates.dat.data[3] += 0.2
        self.assertEqual(checker.check(), 1)
