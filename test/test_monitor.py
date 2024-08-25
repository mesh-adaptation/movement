"""
Unit tests for monitor module.
"""

import unittest

import numpy as np
from firedrake import SpatialCoordinate, UnitSquareMesh, UnitTriangleMesh
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace

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

    def test_tiny_amplitude(self):
        mesh = UnitTriangleMesh()
        mf = BallMonitorFactory(
            dim=2, centre=(0, 0), radius=0.1, amplitude=1e-8, width=0.1
        )
        self.assertTrue(np.allclose(mf.get_monitor()(mesh).dat.data, 1))


class TestGradient(unittest.TestCase):
    """
    Unit tests for :class:`~.GradientMonitorFactory`.
    """

    def test_tiny_scale_factor(self):
        mesh = UnitSquareMesh(5, 5)
        x, y = SpatialCoordinate(mesh)
        sol = Function(FunctionSpace(mesh, "CG", 1)).interpolate(x**2)
        mf = GradientMonitorFactory(dim=2, scale_factor=1e-8, solution=sol)
        self.assertTrue(np.allclose(mf.get_monitor()(mesh).dat.data, 1))

    def test_linear(self):
        mesh = UnitSquareMesh(5, 5)
        x, y = SpatialCoordinate(mesh)
        sol = Function(FunctionSpace(mesh, "CG", 1)).interpolate(x)
        mf = GradientMonitorFactory(dim=2, scale_factor=1, solution=sol)
        self.assertTrue(np.allclose(mf.get_monitor()(mesh).dat.data, 2))


class TestHessian(unittest.TestCase):
    """
    Unit tests for :class:`~.HessianMonitorFactory`.
    """

    def test_tiny_scale_factor(self):
        mesh = UnitSquareMesh(5, 5)
        x, y = SpatialCoordinate(mesh)
        sol = Function(FunctionSpace(mesh, "CG", 1)).interpolate(x**3)
        mf = HessianMonitorFactory(dim=2, scale_factor=1e-8, solution=sol)
        self.assertTrue(np.allclose(mf.get_monitor()(mesh).dat.data, 1))

    def test_quadratic(self):
        mesh = UnitSquareMesh(5, 5)
        x, y = SpatialCoordinate(mesh)
        sol = Function(FunctionSpace(mesh, "CG", 2)).interpolate(0.5 * x**2)
        mf = HessianMonitorFactory(dim=2, scale_factor=1, solution=sol)
        self.assertTrue(np.allclose(mf.get_monitor()(mesh).dat.data, 2))
