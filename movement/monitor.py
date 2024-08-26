import abc

import ufl
from animate.recovery import recover_gradient_l2, recover_hessian_clement
from animate.utility import norm
from firedrake import SpatialCoordinate
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.functionspace import (
    FunctionSpace,
    TensorFunctionSpace,
    VectorFunctionSpace,
)

__all__ = [
    "ConstantMonitorBuilder",
    "BallMonitorBuilder",
    "GradientMonitorBuilder",
    "HessianMonitorBuilder",
    "GradientHessianMonitorBuilder",
]


class MonitorBuilder(metaclass=abc.ABCMeta):
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


class ConstantMonitorBuilder(MonitorBuilder):
    """
    Builder class for constant monitor functions.
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


class BallMonitorBuilder(MonitorBuilder):
    r"""
    Builder class for monitor functions focused around ball shapes:

    .. math::
        m(\mathbf{x}) = 1 + \frac{\alpha}
        {\cosh^2\left(\beta\left((\mathbf{x}-\mathbf{c})\cdot(\mathbf{x}-\mathbf{c})
        -\gamma^2\right)\right)},

    where :math:`\mathbf{c}` is the centre point, :math:`\alpha` is the amplitude of the
    monitor function, :math:`\beta` is the width of the transition region, and
    :math:`\gamma` is the radius of the ball.
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


class SolutionBasedMonitorBuilder(MonitorBuilder, metaclass=abc.ABCMeta):
    """
    Abstract base class for monitor factories based on solution data.
    """

    @abc.abstractmethod
    def __init__(self, dim, solution):
        """
        :arg dim: mesh dimension
        :type dim: :class:`int`
        :arg solution: solution to base the monitor on
        :type solution: :class:`firedrake.function.Function`
        """
        super().__init__(dim)
        assert isinstance(solution, Function)
        self.solution = solution

    def projection(self, mesh):
        """
        Project the solution field onto the given mesh.

        :arg mesh: the mesh on which the solution is to be projected
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :return: the projected solution field
        :rtype: :class:`firedrake.function.Function`
        """
        fs = FunctionSpace(mesh, self.solution.ufl_element())
        return Function(fs).project(self.solution)


# TODO: Support computing gradient with Clement interpolant
class GradientMonitorBuilder(SolutionBasedMonitorBuilder):
    r"""
    Builder class for monitor functions based on gradients of solutions:

    .. math::
        m(\mathbf{x}) = 1 + \alpha\frac{\nabla u\cdot\nabla u}
        {\max_{x\in\Omega}\nabla u\cdot\nabla u},

    where :math:`\alpha` is a scale factor and :math:`u` is the solution field of
    interest.
    """

    def __init__(self, dim, solution, scale_factor):
        """
        :arg dim: mesh dimension
        :type dim: :class:`int`
        :arg solution: solution to recover the gradient of
        :type solution: :class:`firedrake.function.Function`
        :arg scale_factor: scale factor for the gradient part
        :type scale_factor: :class:`float`
        """
        super().__init__(dim, solution)
        assert scale_factor > 0
        self.gradient_scale_factor = Constant(scale_factor)

    def recover_gradient(self, target_space):
        r"""
        Recover the gradient of the solution field projected onto the current mesh.

        :arg target_space: space to recover gradient in
        :type target_space: :class:`firedrake.functionspace.FunctionSpace`
        :return: the recovered gradient in vector :math:`\mathbb{P}1` space
        :rtype: :class:`firedrake.function.Function`
        """
        mesh = target_space.mesh()
        return recover_gradient_l2(self.projection(mesh), target_space=target_space)

    def monitor(self, mesh):
        """
        Monitor function based on recovered gradient.

        :arg mesh: the mesh on which the monitor function is to be defined
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :return: gradient-based monitor function evaluated on given mesh
        :rtype: :class:`firedrake.function.Function`
        """
        g = self.recover_gradient(VectorFunctionSpace(mesh, "CG", 1))
        gg = Function(FunctionSpace(mesh, "CG", 1)).interpolate(ufl.dot(g, g))
        return Constant(1.0) + self.gradient_scale_factor * (
            gg / norm(gg, norm_type="linf")
        )


# TODO: Support computing Hessian with double L2 projection
class HessianMonitorBuilder(SolutionBasedMonitorBuilder):
    r"""
    Builder class for monitor functions based on Hessians of solutions.

    .. math::
        m(\mathbf{x}) = 1 + \alpha\frac{\nabla u\cdot\nabla u}
        {\max_{x\in\Omega}\nabla u\cdot\nabla u} +
        \beta\frac{\mathbf{H}(u):\mathbf{H}(u)}
        {\max_{x\in\Omega}\mathbf{H}(u):\mathbf{H}(u)},

    where :math:`\alpha` is a scale factor, :math:`u` is the solution field of interest,
    and :math:`\mathbf{H}(u)` is the Hessian
    """

    def __init__(self, dim, solution, scale_factor):
        """
        :arg dim: mesh dimension
        :type dim: :class:`int`
        :arg solution: solution to recover the Hessian of
        :type solution: :class:`firedrake.function.Function`
        :arg scale_factor: scale factor for the Hessian part
        :type scale_factor: :class:`float`
        """
        super().__init__(dim, solution)
        assert scale_factor > 0
        self.hessian_scale_factor = Constant(scale_factor)

    def recover_hessian(self, target_space):
        r"""
        Recover the Hessian of the solution field.

        :arg target_space: space to recover Hessian in
        :type target_space: :class:`firedrake.functionspace.FunctionSpace`
        :return: the recovered Hessian in tensor :math:`\mathbb{P}1` space
        :rtype: :class:`firedrake.function.Function`
        """
        return recover_hessian_clement(self.projection(target_space.mesh()))[1]

    def monitor(self, mesh):
        """
        Monitor function based on recovered Hessian.

        :arg mesh: the mesh on which the monitor function is to be defined
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :return: Hessian-based monitor function evaluated on given mesh
        :rtype: :class:`firedrake.function.Function`
        """
        H = self.recover_hessian(TensorFunctionSpace(mesh, "CG", 1))
        frob = sum(H[i, j] ** 2 for i in range(self.dim) for j in range(self.dim))
        HH = Function(FunctionSpace(mesh, "CG", 1)).interpolate(frob)
        return Constant(1.0) + self.hessian_scale_factor * (
            HH / norm(HH, norm_type="linf")
        )


class GradientHessianMonitorBuilder(GradientMonitorBuilder, HessianMonitorBuilder):
    r"""
    Builder class for monitor functions based on both gradients and Hessians of
    solutions.

    .. math::
        m(\mathbf{x}) = 1 + \alpha\frac{\nabla u\cdot\nabla u}
        {\max_{x\in\Omega}\nabla u\cdot\nabla u} +
        \beta\frac{\mathbf{H}(u):\mathbf{H}(u)}
        {\max_{x\in\Omega}\mathbf{H}(u):\mathbf{H}(u)},

    where :math:`\alpha` is a scale factor for the gradient part, :math:`\beta` is a
    scale factor for the Hessian part, :math:`u` is the solution field of interest, and
    :math:`\mathbf{H}(u)` is the Hessian
    """

    def __init__(self, dim, solution, gradient_scale_factor, hessian_scale_factor):
        """
        :arg dim: mesh dimension
        :type dim: :class:`int`
        :arg gradient_scale_factor: scale factor for the gradient part
        :type gradient_scale_factor: :class:`float`
        :arg hessian_scale_factor: scale factor for the Hessian part
        :type hessian_scale_factor: :class:`float`
        :arg solution: solution to recover the gradient and Hessian of
        :type solution: :class:`firedrake.function.Function`
        """
        SolutionBasedMonitorBuilder.__init__(self, dim, solution)
        self.gradient_scale_factor = Constant(gradient_scale_factor)
        self.hessian_scale_factor = Constant(hessian_scale_factor)

    def monitor(self, mesh):
        """
        Monitor function based on recovered gradient and Hessian.

        :arg mesh: the mesh on which the monitor function is to be defined
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :return: gradient and Hessian-based monitor function evaluated on given mesh
        :rtype: :class:`firedrake.function.Function`
        """
        # Recover gradient
        g = self.recover_gradient(VectorFunctionSpace(mesh, "CG", 1))
        gg = Function(FunctionSpace(mesh, "CG", 1)).interpolate(ufl.dot(g, g))

        # Recover Hessian
        H = self.recover_hessian(TensorFunctionSpace(mesh, "CG", 1))
        frob = sum(H[i, j] ** 2 for i in range(self.dim) for j in range(self.dim))
        HH = Function(FunctionSpace(mesh, "CG", 1)).interpolate(frob)

        # Combine both gradient and Hessian parts
        return (
            Constant(1.0)
            + self.gradient_scale_factor * (gg / norm(gg, norm_type="linf"))
            + self.hessian_scale_factor * (HH / norm(HH, norm_type="linf"))
        )
