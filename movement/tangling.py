import warnings

import firedrake
import ufl
from firedrake.__future__ import interpolate

__all__ = ["MeshTanglingChecker"]


class MeshTanglingChecker:
    """
    A class for tracking whether a mesh has tangled, i.e. whether any of its elements
    have become inverted.
    """

    def __init__(self, mesh, raise_error=True):
        """
        :kwarg mesh: the mesh to track if tangled
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg raise_error: if ``True``, an error is raised if any element is tangled,
            otherwise a warning is raised
        :type raise_error: :class:`bool`
        """
        self.mesh = mesh
        self.raise_error = raise_error
        P0 = firedrake.FunctionSpace(mesh, "DG", 0)
        self._jacobian_sign = firedrake.Function(P0)
        detJ = ufl.JacobianDeterminant(mesh)
        s = firedrake.assemble(interpolate(ufl.sign(detJ), P0))
        self._jacobian_sign_expr = ufl.sign(detJ) / s

    @property
    def scaled_jacobian(self):
        return self._jacobian_sign.interpolate(self._jacobian_sign_expr)

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
