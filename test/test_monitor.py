"""
Unit tests for monitor module.
"""

import unittest

import numpy as np
from firedrake import UnitTriangleMesh

from movement.monitor import *


class TestConstant(unittest.TestCase):
    """
    Unit tests for :class:`~.ConstantMonitorFactory`.
    """

    def test_value(self):
        mesh = UnitTriangleMesh()
        mf = ConstantMonitorFactory(mesh.topological_dimension())
        self.assertTrue(np.allclose(mf.monitor(mesh).dat.data, 1))
