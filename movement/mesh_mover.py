import firedrake


__all__ = ["MeshMover"]


class MeshMover(object):
    """
    Base class for all mesh movers.
    """
    def __init__(self, mesh, monitor_function=None, **kwargs):
        self.mesh = mesh
        self.monitor_function = monitor_function
        self.dim = mesh.topological_dimension()
        if self.dim != 2:
            raise NotImplementedError(f"Dimension {self.dim} has not been considered yet")

        # Measures
        degree = kwargs.get('quadrature_degree')
        self.dx = firedrake.dx(domain=mesh, degree=degree)
        self.ds = firedrake.ds(domain=mesh, degree=degree)
        self.dS = firedrake.dS(domain=mesh, degree=degree)

        # Mesh coordinate functions
        self._x = firedrake.Function(self.mesh.coordinates, name="Physical coordinates")
        self.xi = firedrake.Function(self.mesh.coordinates, name="Computational coordinates")

    def adapt(self):
        raise NotImplementedError("Implement `adapt` in the derived class.")
