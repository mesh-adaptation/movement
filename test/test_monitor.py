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
        self.assertTrue(np.allclose(mf.get_monitor()(mesh).dat.data, 1))


class TestBall(unittest.TestCase):
    """
    Unit tests for :class:`~.BallMonitorFactory`.
    """

    def test_zero_amplitude(self):
        mesh = UnitTriangleMesh()
        mf = BallMonitorFactory(
            dim=2, centre=(0, 0), radius=0.1, amplitude=1e-8, width=0.1
        )
        self.assertTrue(np.allclose(mf.get_monitor()(mesh).dat.data, 1))


class TestGradient(unittest.TestCase):
    """
    Unit tests for :class:`~.GradientMonitorFactory`.
    """

    pass  # TODO


class TestHessian(unittest.TestCase):
    """
    Unit tests for :class:`~.HessianMonitorFactory`.
    """

    pass  # TODO
