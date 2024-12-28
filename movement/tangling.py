import abc
import warnings

import firedrake
import ufl
from firedrake.__future__ import interpolate

__all__ = ["MeshTanglingChecker"]


class MeshTanglingChecker_Base(abc.ABC):
    """
    Base class for tracking whether a mesh has tangled, i.e. whether any of its elements
    have become inverted.
    """

    @abc.abstractmethod
    def __init__(self, mesh, raise_error):
        """
        :arg mesh: the mesh to track if tangled
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg raise_error: if ``True``, an error is raised if any element is tangled,
            otherwise a warning is raised
        :type raise_error: :class:`bool`
        """
        self.mesh = mesh
        self.raise_error = raise_error
        self.P0 = firedrake.FunctionSpace(mesh, "DG", 0)
        self._sj = firedrake.Function(self.P0)
        self._sj_expr = None

    @property
    def scaled_jacobian(self):
        assert self._sj_expr is not None
        self._sj.interpolate(self._sj_expr)
        return self._sj

    def check(self):
        """
        Check whether any element orientations have changed since the tangling checker
        was created.
        """
        sj = self.scaled_jacobian.dat.data_with_halos
        num_tangled = len(sj[sj < 0])
        if num_tangled > 0:
            plural = "s" if num_tangled > 1 else ""
            msg = f"Mesh has {num_tangled} tangled element{plural}."
            if self.raise_error:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=1)
        return num_tangled


class MeshTanglingChecker_1D(MeshTanglingChecker_Base):
    """
    Implementation of mesh tangling check in 1D.
    """

    def __init__(self, mesh, raise_error=True):
        """
        :arg mesh: the mesh to track if tangled
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :kwarg raise_error: if ``True``, an error is raised if any element is tangled,
            otherwise a warning is raised
        :type raise_error: :class:`bool`
        """
        super().__init__(mesh, raise_error)
        detJ = ufl.JacobianDeterminant(mesh)
        s = firedrake.assemble(interpolate(ufl.sign(detJ), self.P0))
        self._sj_expr = ufl.sign(detJ) / s


class MeshTanglingChecker_2D(MeshTanglingChecker_Base):
    """
    Implementation of mesh tangling check in 2D.
    """

    def __init__(self, mesh, raise_error=True):
        """
        :arg mesh: the mesh to track if tangled
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :kwarg raise_error: if ``True``, an error is raised if any element is tangled,
            otherwise a warning is raised
        :type raise_error: :class:`bool`
        """
        super().__init__(mesh, raise_error)
        detJ = ufl.JacobianDeterminant(mesh)
        s = firedrake.assemble(interpolate(ufl.sign(detJ), self.P0))
        self._sj_expr = ufl.sign(detJ) / s


def MeshTanglingChecker(mesh, **kwargs):
    """
    Factory function for creating mesh tangling checker classes.

    Note that the implementation is based on element orientations, so a reflection in
    the :math:`x` coordinate (for example) will report mesh tangling, even though the
    mesh doesn't 'look' tangled.

    :arg mesh: the mesh to track if tangled
    :type mesh: :class:`firedrake.mesh.MeshGeometry`
    :kwarg raise_error: if ``True``, an error is raised if any element is tangled,
        otherwise a warning is raised
    :type raise_error: :class:`bool`
    """
    dim = mesh.topological_dimension()
    try:
        implementations = {
            1: MeshTanglingChecker_1D,
            2: MeshTanglingChecker_2D,
        }
        return implementations[dim](mesh, **kwargs)
    except KeyError as ke:
        raise ValueError(f"Mesh dimension {dim} not supported.") from ke
