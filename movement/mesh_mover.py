import firedrake


__all__ = ["MeshMover"]


class MeshMover(object):
    """
    Base class for all mesh movers.
    """
    def __init__(self, mesh, monitor_function=None):
        self.mesh = mesh
        self.monitor_function = monitor_function

        self.dim = mesh.topological_dimension()
        if self.dim != 2:
            raise NotImplementedError(f"Dimension {self.dim} has not been considered yet")

        self.x = firedrake.Function(self.mesh.coordinates, name="Physical coordinates")
        self.xi = firedrake.Function(self.mesh.coordinates, name="Computational coordinates")

    def adapt(self):
        raise NotImplementedError("Implement `adapt` in the derived class.")
