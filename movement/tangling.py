import warnings

import firedrake
import ufl
from firedrake.petsc import PETSc

__all__ = ["MeshTanglingChecker"]


class MeshTanglingChecker:
    """
    A class for tracking whether a mesh has tangled, i.e. whether any of its elements
    have become inverted.
    """

    def __init__(self, mesh, raise_error=True):
        """
        :arg mesh: the mesh to track if tangled
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :kwarg raise_error: if ``True``, an error is raised if any element is tangled,
            otherwise a warning is raised
        :type raise_error: :class:`bool`
        """
        self.mesh = mesh
        self.raise_error = raise_error
        P0 = firedrake.FunctionSpace(mesh, "DG", 0)
        detJ = ufl.JacobianDeterminant(mesh)
        self._detJ_ratio_expr = detJ / firedrake.Function(P0).interpolate(detJ)
        self._detJ_ratio = firedrake.Function(P0)

    @property
    def jacobian_determinant_ratio(self):
        """
        Compute the ratio of the determinant of the Jacobian of the current mesh to
        the determinant of the Jacobian of the original mesh.
        """
        return self._detJ_ratio.interpolate(self._detJ_ratio_expr)

    @PETSc.Log.EventDecorator()
    def check(self):
        """
        Check whether any element orientations have changed since the tangling checker
        was created.
        """
        jdr = self.jacobian_determinant_ratio.dat.data_with_halos
        num_tangled = len(jdr[jdr <= 0])
        if num_tangled > 0:
            plural = "s" if num_tangled > 1 else ""
            msg = f"Mesh has {num_tangled} tangled element{plural}."
            if self.raise_error:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=1)
        return num_tangled
