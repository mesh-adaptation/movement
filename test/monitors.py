from firedrake import *


def const_monitor(mesh):
    return Constant(1.0)


def reciprocal_monitor(f):
    r"""Monitor :math:`m(xi) = 1/((f-1)\xi+1)`

    s.t. :math:`m(0)=1` and :math:`m(1)=1/f` and which corresponds
    to :math:`X(\xi) = a\xi^2 +b\xi` with :math:`a=(f-1)/(f+1)` and :math:`b=2/(f+1)`,
    and :math:`X(0) = 0` and :math:`X(1) = 1`.

    In multiple dimensions :math:`m(xi) = \prod_i 1/((f_i)/\xi_i + 1)`"""
    a = [(fi - 1) / (fi + 1) for fi in f]
    b = [2 / (fi + 1) for fi in f]

    def monitor(mesh):
        xy = SpatialCoordinate(mesh)
        xi = [
            (-bi + sqrt(bi**2 + 4 * ai * x_i)) / 2 / ai for ai, bi, x_i in zip(a, b, xy)
        ]
        return product(1 / ((fi - 1) * xi_i + 1) for fi, xi_i in zip(f, xi))

    return monitor


def ring_monitor(mesh):
    alpha = Constant(10.0)  # amplitude
    beta = Constant(200.0)  # width
    gamma = Constant(0.15)  # radius
    dim = mesh.geometric_dimension()
    xyz = SpatialCoordinate(mesh) - as_vector([0.5] * dim)
    r = dot(xyz, xyz)
    return Constant(1.0) + alpha / cosh(beta * (r - gamma)) ** 2
