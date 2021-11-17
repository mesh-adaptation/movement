import firedrake
import ufl


__all__ = ["MeshTanglingChecker"]


class MeshTanglingChecker(object):
    """
    A class for tracking whether
    """
    def __init__(self, mesh, mode='warn'):
        """
        :arg mesh: the mesh to track if tangled
        :kwarg mode: should a warning or an error
            be raised when tangling is encountered?
        """
        self.mesh = mesh
        if mode not in ['warn', 'error']:
            raise ValueError(f"Choose mode from 'warn' and 'error'")
        self.mode = mode

        # Store initial signs of Jacobian determinant
        P0 = firedrake.FunctionSpace(mesh, 'DG', 0)
        detJ = ufl.JacobianDeterminant(mesh)
        s = firedrake.Function(P0)
        s.interpolate(ufl.sign(detJ))

        # Get scaled Jacobian expression
        P0_ten = firedrake.TensorFunctionSpace(mesh, 'DG', 0)
        J = firedrake.interpolate(ufl.Jacobian(mesh), P0_ten)
        edge1 = ufl.as_vector([J[0, 0], J[1, 0]])
        edge2 = ufl.as_vector([J[0, 1], J[1, 1]])
        edge3 = edge1 - edge2
        norm1 = ufl.sqrt(ufl.dot(edge1, edge1))
        norm2 = ufl.sqrt(ufl.dot(edge2, edge2))
        norm3 = ufl.sqrt(ufl.dot(edge3, edge3))
        prod1 = ufl.max_value(norm1*norm2, norm1*norm3)
        prod2 = ufl.max_value(norm2*norm3, norm2*norm1)
        prod3 = ufl.max_value(norm3*norm1, norm3*norm2)
        maxval = ufl.max_value(ufl.max_value(prod1, prod2), prod3)
        self._sj_expr = detJ/maxval*s
        self._sj = firedrake.Function(P0)

    @property
    def scaled_jacobian(self):
        self._sj.interpolate(self._sj_expr)
        return self._sj

    def check(self):
        """
        Check for tangling.
        """
        sj = self.scaled_jacobian.dat.data
        num_tangled = len(sj[sj < 0])
        if num_tangled == 0:
            return 0
        msg = f'Mesh has {num_tangled} tangled elements'
        if self.mode == 'warn':
            import warnings
            warnings.warn(msg)
        else:
            raise ValueError(msg)
        return num_tangled
