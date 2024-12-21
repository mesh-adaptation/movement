import unittest
from unittest.mock import MagicMock

from firedrake.utility_meshes import UnitSquareMesh

from movement import LaplacianSmoother


class TestExceptions(unittest.TestCase):
    """
    Unit tests for exceptions raised during Laplacian smoothing.
    """

    def test_tangling_valueerror(self):
        mover = LaplacianSmoother(UnitSquareMesh(3, 3), 1.0)
        mover.x.dat.data[3] += 0.2
        mover._solve = MagicMock()
        with self.assertRaises(ValueError) as cm:
            mover.move(0)
        self.assertEqual(str(cm.exception), "Mesh has 1 tangled element.")
