"""
Unit tests for monitor module.
"""

import unittest

import numpy as np
from firedrake import SpatialCoordinate, UnitSquareMesh
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.norms import errornorm

from movement.monitor import (
    BallMonitorBuilder,
    ConstantMonitorBuilder,
    GradientHessianMonitorBuilder,
    GradientMonitorBuilder,
    HessianMonitorBuilder,
)


class BaseClasses:
    """
    Base classes for monitor factories.
    """

    class TestMonitorBuilder(unittest.TestCase):
        """
        Base class for monitor factory unit tests.
        """

        def setUp(self):
            self.mesh = UnitSquareMesh(5, 5)
            self.P1 = FunctionSpace(self.mesh, "CG", 1)
            self.solution = Function(self.P1)


class TestConstant(BaseClasses.TestMonitorBuilder):
    """
    Unit tests for :class:`~.ConstantMonitorBuilder`.
    """

    def test_value_get_monitor(self):
        mb = ConstantMonitorBuilder(self.mesh.topological_dimension())
        self.assertTrue(np.allclose(mb.get_monitor()(self.mesh).dat.data, 1))

    def test_value_call(self):
        mb = ConstantMonitorBuilder(self.mesh.topological_dimension())
        self.assertTrue(np.allclose(mb()(self.mesh).dat.data, 1))


class TestBall(BaseClasses.TestMonitorBuilder):
    """
    Unit tests for :class:`~.BallMonitorBuilder`.
    """

    def test_tiny_amplitude(self):
        mb = BallMonitorBuilder(
            dim=2, centre=(0, 0), radius=0.1, amplitude=1e-8, width=0.1
        )
        self.assertTrue(np.allclose(mb.get_monitor()(self.mesh).dat.data, 1))


class TestGradient(BaseClasses.TestMonitorBuilder):
    """
    Unit tests for :class:`~.GradientMonitorBuilder`.
    """

    def test_tiny_scale_factor(self):
        x, y = SpatialCoordinate(self.mesh)
        self.solution.interpolate(x**2)
        mb = GradientMonitorBuilder(dim=2, solution=self.solution, scale_factor=1e-8)
        self.assertTrue(np.allclose(mb.get_monitor()(self.mesh).dat.data, 1))

    def test_linear(self):
        x, y = SpatialCoordinate(self.mesh)
        self.solution.interpolate(x)
        mb = GradientMonitorBuilder(dim=2, solution=self.solution, scale_factor=1)
        self.assertTrue(np.allclose(mb.get_monitor()(self.mesh).dat.data, 2))


class TestHessian(BaseClasses.TestMonitorBuilder):
    """
    Unit tests for :class:`~.HessianMonitorBuilder`.
    """

    def test_tiny_scale_factor(self):
        x, y = SpatialCoordinate(self.mesh)
        self.solution.interpolate(x**3)
        mb = HessianMonitorBuilder(dim=2, solution=self.solution, scale_factor=1e-8)
        self.assertTrue(np.allclose(mb.get_monitor()(self.mesh).dat.data, 1))

    def test_quadratic(self):
        x, y = SpatialCoordinate(self.mesh)
        P2 = FunctionSpace(self.mesh, "CG", 2)
        solution = Function(P2)
        solution.interpolate(0.5 * x**2)
        mb = HessianMonitorBuilder(dim=2, solution=solution, scale_factor=1)
        self.assertTrue(np.allclose(mb.get_monitor()(self.mesh).dat.data, 2))


class TestGradientHessian(BaseClasses.TestMonitorBuilder):
    """
    Unit tests for :class:`~.GradientHessianMonitorBuilder`.
    """

    def test_tiny_scale_factors(self):
        x, y = SpatialCoordinate(self.mesh)
        self.solution.interpolate(x**3)
        mb = GradientHessianMonitorBuilder(
            dim=2,
            solution=self.solution,
            gradient_scale_factor=1e-8,
            hessian_scale_factor=1e-8,
        )
        self.assertTrue(np.allclose(mb.get_monitor()(self.mesh).dat.data, 1))

    def test_tiny_hessian_scale_factor(self):
        x, y = SpatialCoordinate(self.mesh)
        self.solution.interpolate(x**3)
        mb1 = GradientHessianMonitorBuilder(
            dim=2,
            solution=self.solution,
            gradient_scale_factor=1,
            hessian_scale_factor=1e-8,
        )
        mb2 = GradientMonitorBuilder(
            dim=2,
            solution=self.solution,
            scale_factor=1,
        )
        self.assertAlmostEqual(
            errornorm(mb1.get_monitor()(self.mesh), mb2.get_monitor()(self.mesh)), 0
        )

    def test_tiny_gradient_scale_factor(self):
        x, y = SpatialCoordinate(self.mesh)
        self.solution.interpolate(x**3)
        mb1 = GradientHessianMonitorBuilder(
            dim=2,
            solution=self.solution,
            gradient_scale_factor=1e-8,
            hessian_scale_factor=1,
        )
        mb2 = HessianMonitorBuilder(
            dim=2,
            solution=self.solution,
            scale_factor=1,
        )
        self.assertAlmostEqual(
            errornorm(mb1.get_monitor()(self.mesh), mb2.get_monitor()(self.mesh)), 0
        )
