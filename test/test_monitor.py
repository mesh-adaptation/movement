"""
Unit tests for monitor module.
"""

import unittest

import numpy as np
from firedrake import SpatialCoordinate, UnitSquareMesh
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.norms import errornorm

from movement.monitor import *


class BaseClasses:
    """
    Base classes for monitor factories.
    """

    class TestMonitorFactory(unittest.TestCase):
        """
        Base class for monitor factory unit tests.
        """

        def setUp(self):
            self.mesh = UnitSquareMesh(5, 5)
            self.P1 = FunctionSpace(self.mesh, "CG", 1)
            self.solution = Function(self.P1)


class TestConstant(BaseClasses.TestMonitorFactory):
    """
    Unit tests for :class:`~.ConstantMonitorFactory`.
    """

    def test_value(self):
        mf = ConstantMonitorFactory(self.mesh.topological_dimension())
        self.assertTrue(np.allclose(mf.get_monitor()(self.mesh).dat.data, 1))


class TestBall(BaseClasses.TestMonitorFactory):
    """
    Unit tests for :class:`~.BallMonitorFactory`.
    """

    def test_tiny_amplitude(self):
        mf = BallMonitorFactory(
            dim=2, centre=(0, 0), radius=0.1, amplitude=1e-8, width=0.1
        )
        self.assertTrue(np.allclose(mf.get_monitor()(self.mesh).dat.data, 1))


class TestGradient(BaseClasses.TestMonitorFactory):
    """
    Unit tests for :class:`~.GradientMonitorFactory`.
    """

    def test_tiny_scale_factor(self):
        x, y = SpatialCoordinate(self.mesh)
        self.solution.interpolate(x**2)
        mf = GradientMonitorFactory(dim=2, solution=self.solution, scale_factor=1e-8)
        self.assertTrue(np.allclose(mf.get_monitor()(self.mesh).dat.data, 1))

    def test_linear(self):
        x, y = SpatialCoordinate(self.mesh)
        self.solution.interpolate(x)
        mf = GradientMonitorFactory(dim=2, solution=self.solution, scale_factor=1)
        self.assertTrue(np.allclose(mf.get_monitor()(self.mesh).dat.data, 2))


class TestHessian(BaseClasses.TestMonitorFactory):
    """
    Unit tests for :class:`~.HessianMonitorFactory`.
    """

    def test_tiny_scale_factor(self):
        x, y = SpatialCoordinate(self.mesh)
        self.solution.interpolate(x**3)
        mf = HessianMonitorFactory(dim=2, solution=self.solution, scale_factor=1e-8)
        self.assertTrue(np.allclose(mf.get_monitor()(self.mesh).dat.data, 1))

    def test_quadratic(self):
        x, y = SpatialCoordinate(self.mesh)
        P2 = FunctionSpace(self.mesh, "CG", 2)
        solution = Function(P2)
        solution.interpolate(0.5 * x**2)
        mf = HessianMonitorFactory(dim=2, solution=solution, scale_factor=1)
        self.assertTrue(np.allclose(mf.get_monitor()(self.mesh).dat.data, 2))


class TestGradientHessian(BaseClasses.TestMonitorFactory):
    """
    Unit tests for :class:`~.GradientHessianMonitorFactory`.
    """

    def test_tiny_scale_factors(self):
        x, y = SpatialCoordinate(self.mesh)
        self.solution.interpolate(x**3)
        mf = GradientHessianMonitorFactory(
            dim=2,
            solution=self.solution,
            gradient_scale_factor=1e-8,
            hessian_scale_factor=1e-8,
        )
        self.assertTrue(np.allclose(mf.get_monitor()(self.mesh).dat.data, 1))

    def test_tiny_hessian_scale_factor(self):
        x, y = SpatialCoordinate(self.mesh)
        self.solution.interpolate(x**3)
        mf1 = GradientHessianMonitorFactory(
            dim=2,
            solution=self.solution,
            gradient_scale_factor=1,
            hessian_scale_factor=1e-8,
        )
        mf2 = GradientMonitorFactory(
            dim=2,
            solution=self.solution,
            scale_factor=1,
        )
        self.assertAlmostEqual(
            errornorm(mf1.get_monitor()(self.mesh), mf2.get_monitor()(self.mesh)), 0
        )

    def test_tiny_gradient_scale_factor(self):
        x, y = SpatialCoordinate(self.mesh)
        self.solution.interpolate(x**3)
        mf1 = GradientHessianMonitorFactory(
            dim=2,
            solution=self.solution,
            gradient_scale_factor=1e-8,
            hessian_scale_factor=1,
        )
        mf2 = HessianMonitorFactory(
            dim=2,
            solution=self.solution,
            scale_factor=1,
        )
        self.assertAlmostEqual(
            errornorm(mf1.get_monitor()(self.mesh), mf2.get_monitor()(self.mesh)), 0
        )
