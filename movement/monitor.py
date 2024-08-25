import abc

import ufl
from animate.recovery import recover_gradient_l2
from animate.utility import norm
from firedrake import SpatialCoordinate
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace

__all__ = [
    "ConstantMonitorFactory",
    "BallMonitorFactory",
    "GradientMonitorFactory",
    "HessianMonitorFactory",
]


class MonitorFactory(metaclass=abc.ABCMeta):
    """
    Abstract base class for monitor function factories.
    """

    def __init__(self, dim):
        """
        :arg dim: dimension of the mesh
        :type dim: :class:`int`
        """
        self.dim = dim

    @abc.abstractmethod
    def monitor(self, mesh):
        """
        Abstract method to create a monitor function.

        :arg mesh: the mesh on which the monitor function is to be defined
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :return: monitor function evaluated on given mesh
        :rtype: :class:`firedrake.function.Function`
        """
        pass

    def get_monitor(self):
        """
        Returns a callable monitor function whose only argument is the mesh.

        :return: monitor function
        :rtype: callable monitor function with a single argument
        """

        def monitor(mesh):
            m = self.monitor(mesh)
            if not isinstance(m, (Constant, Function)):
                m = Function(FunctionSpace(mesh, "CG", 1)).interpolate(m)
            return m

        return monitor

    def __call__(self):
        """
        Alias for :meth:`get_monitor`.

        :return: monitor function
        :rtype: callable monitor function with a single argument
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
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :return: constant monitor function
        :rtype: :class:`firedrake.function.Function`
        """
        return Constant(1.0)


class BallMonitorFactory(MonitorFactory):
    """
    Factory class for monitor functions focused around ball shapes.
    """

    def __init__(self, dim, centre, radius, amplitude, width):
        r"""
        :arg dim: mesh dimension
        :type dim: :class:`int`
        :arg centre: the centre of the ball
        :type centre: :class:`tuple` of :class:`float`\s
        :arg radius: the radius of the ball
        :type radius: :class:`float`
        :arg amplitude: the amplitude of the monitor function
        :type amplitude: :class:`float`
        :arg width: the width of the transition region
        :type width: :class:`float`
        """
        super().__init__(dim)
        assert len(centre) == self.dim
        assert radius > 0
        assert amplitude > 0
        assert width > 0
        self.centre = ufl.as_vector([Constant(c) for c in centre])
        self.radius = Constant(radius)
        self.amplitude = Constant(amplitude)
        self.width = Constant(width)

    def monitor(self, mesh):
        """
        Creates a monitor function focused around a ball shape.

        :arg mesh: the mesh on which the monitor function is to be defined
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :return: ball-shaped monitor function
        :rtype: :class:`firedrake.function.Function`
        """
        diff = SpatialCoordinate(mesh) - self.centre
        dist = ufl.dot(diff, diff)
        return (
            Constant(1.0)
            + self.amplitude / ufl.cosh(self.width * (dist - self.radius**2)) ** 2
        )


class GradientMonitorFactory(MonitorFactory):
    """
    Factory class for monitor functions based on gradients of solutions.
    """

    def __init__(self, dim, scale_factor, solution):
        """
        :arg dim: mesh dimension
        :type dim: :class:`int`
        :arg scale_factor: scale factor for the gradient part
        :type scale_factor: :class:`float`
        :arg solution: solution to recover the gradient of
        :type solution: :class:`firedrake.function.Function`
        """
        super().__init__(dim)
        assert scale_factor > 0
        self.scale_factor = Constant(scale_factor)
        assert isinstance(solution, Function)
        self.solution = solution

    def monitor(self, mesh):
        P1 = FunctionSpace(mesh, "CG", 1)
        P1_vec = VectorFunctionSpace(mesh, "CG", 1)
        g = recover_gradient_l2(self.solution, target_space=P1_vec)
        return Constant(1.0) + self.scale_factor * Function(P1).interpolate(
            ufl.dot(g, g) / norm(g, norm_type="linf")
        )


class HessianMonitorFactory(MonitorFactory):
    """
    Factory class for monitor functions based on Hessians of solutions.
    """

    def monitor(self, mesh):
        raise NotImplementedError  # TODO
