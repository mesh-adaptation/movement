import abc

import ufl
from firedrake import SpatialCoordinate
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.functionspaceimpl import FunctionSpace

__all__ = ["ConstantMonitorFactory", "BallMonitorFactory"]


class MonitorFactory(metaclass=abc.ABCMeta):
    """
    Abstract base class for monitor function factories.
    """

    def __init__(self, dim):
        """
        :arg dim: dimension of the mesh
        """
        self.dim = dim

    @abc.abstractmethod
    def monitor(self, mesh):
        """
        Abstract method to create a monitor function.

        :arg mesh: the mesh on which the monitor function is to be defined
        """
        pass

    def get_monitor(self):
        """
        Returns a callable monitor function whose only argument is the mesh.

        :return: callable monitor function with a single argument
        """

        def monitor(mesh):
            m = self.monitor(mesh)
            if isinstance(m, Constant):
                m = Function(FunctionSpace(mesh, "CG", 1)).interpolate(m)
            if (m.ufl_element().family(), m.ufl_element().degree()) != ("Lagrange", 1):
                raise ValueError("Monitor function must live in P1 space.")
            return

        return monitor

    def __call__(self):
        """
        Alias for :meth:`get_monitor`.
        """
        return self.get_monitor()


class ConstantMonitorFactory(MonitorFactory):
    """
    Factory class for constant monitor functions.
    """

    def monitor(self, mesh):
        """
        Creates a constant monitor function.

        :arg mesh: the mesh on which the monitor function is to be defined
        :return: constant monitor function
        """
        return Constant(1.0)


class BallMonitorFactory(MonitorFactory):
    """
    Factory class for monitor functions focused around ball shapes.
    """

    def __init__(self, dim, centre, radius, amplitude, width):
        """
        :arg dim: mesh dimension
        :arg centre: the centre of the ball
        :arg radius: the radius of the ball
        :arg amplitude: the amplitude of the monitor function
        :arg width: the width of the transition region
        """
        super().__init__(dim)
        assert len(centre) == self.dim
        self.centre = Constant(centre)
        self.radius = Constant(radius)
        self.amplitude = Constant(amplitude)
        self.width = Constant(width)

    def monitor(self, mesh):
        """
        Creates a monitor function focused around a ball shape.

        :arg mesh: the mesh on which the monitor function is to be defined
        :return: ball-shaped monitor function
        """
        diff = SpatialCoordinate(mesh) - self.centre
        dist = ufl.dot(diff, diff)
        return (
            Constant(1.0)
            + self.amplitude / ufl.cosh(self.width * (dist - self.radius)) ** 2
        )
