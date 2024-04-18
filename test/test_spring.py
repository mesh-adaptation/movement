import unittest

import firedrake
import numpy as np

from movement import SpringMover


class TestQuantities(unittest.TestCase):
    """
    Unit tests for the element quantity utilities underlying the :class:`~.SpringMover`
    classes.
    """

    def setUp(self):
        self.mover = SpringMover(firedrake.UnitTriangleMesh(), 1.0)

    @staticmethod
    def rad2deg(radians):
        return 180 * radians / np.pi

    def test_facet_areas(self):
        """
        Test that the edge lenths of a right-angled triangle are computed correctly.
        """
        facet_areas = self.mover.facet_areas.dat.data
        self.assertTrue(np.allclose(facet_areas, [1, np.sqrt(2), 1]))

    def test_tangents(self):
        """
        Test that the tangent vectors of a right-angled triangle are computed correctly.
        """
        tangents = self.mover.tangents.dat.data
        expected = [[1, 0], [np.sqrt(2) / 2, np.sqrt(2) / 2], [0, 1]]
        self.assertTrue(np.allclose(np.abs(tangents), expected))

    def test_angles(self):
        """
        Test that the arguments (angles from the x-axis) of a right-angled triangle are
        computed correctly.
        """
        angles = self.rad2deg(self.mover.angles.dat.data)
        self.assertTrue(np.allclose(angles, [0, 135, 90]))
