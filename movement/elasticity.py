from movement.mover import PrimeMover

__all__ = ["ElasticMover", "LinearElasticMover", "NonlinearElasticMover"]


def ElasticMover(mesh, nonlinear=False, **kwargs):
    """
    Factory function for creating Movers based on elasticity equations.
    """
    raise NotImplementedError  # TODO (#32)


# TODO: Make abc
class ElasticMover_Base(PrimeMover):
    """
    Base class for Movers based on elasticity equations.
    """

    raise NotImplementedError  # TODO (#32)


class LinearElasticMover(ElasticMover_Base):
    r"""
    This Mover is based on the linear elasticity equations,

    .. math::
        \left\{\begin{array}{rl}
        \nabla_{\mathbf{X}}\cdot\sigma(\mathbf{x}) = 0 & \mathbf{x}\in\Omega\\
        \mathbf{x}=\mathbf{x}_D & \mathbf{x}\in\partial\Omega
        \end{array}\right.,

    where :math:`\sigma` is a symmetric displacement-based stress tensor,

    .. math::
        \sigma(\mathbf{x}) =
            \lambda\nabla_{\mathbf{X}}\cdot\mathbf{x}\underline{\mathbf{I}}
            + \mu(\nabla_{\mathbf{X}}\mathbf{x}+(\nabla_{\mathbf{X}}\mathbf{x})^T).

    Here :math:`\lambda` and :math:`\mu` are user-specified Lamé parameters, which
    express the resistance of the mesh to compression and shearing, respectively.

    Since :math:`\mathbf{X}` is fixed in time, the above may be implemented in terms of
    a mesh velocity :math:`\mathbf{v}` as follows:

    .. math::
        \left\{\begin{array}{r}
        \nabla_{\mathbf{X}}\cdot\dot{\sigma}(\mathbf{v}) = 0\\
        \dot{\sigma}(\mathbf{v}) =
            \lambda\nabla_{\mathbf{X}}\cdot\mathbf{v}\underline{\mathbf{I}}
            + \mu(\nabla_{\mathbf{X}}\mathbf{v}+(\nabla_{\mathbf{X}}\mathbf{v})^T)
        \end{array}\right..

    """

    raise NotImplementedError  # TODO (#32)


class NonlinearElasticMover(ElasticMover_Base):
    r"""
    This Mover is based on the nonlinear elasticity equations...
    """

    raise NotImplementedError  # TODO (#34)
