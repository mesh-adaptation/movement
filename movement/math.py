import numpy as np
import sympy

__all__ = []


def equation_of_line(a, b):
    """
    Deduce an expression for the equation of a line passing through two points.

    :arg a: the first point the line passes through
    :type a: :class:`tuple`
    :arg b: the second point the line passes through
    :type b: :class:`tuple`
    :returns: a function of two variables representing the line
    :rtype: :class:`~.Callable`
    """
    assert len(a) == 2
    x0, y0 = a
    assert len(b) == 2
    x1, y1 = b

    if np.isclose(x0, x1):
        # Special case of a vertical line

        def f(x, y):
            return x - x0

    else:

        def f(x, y):
            m = (y1 - y0) / (x1 - x0)
            c = y0 - m * x0
            return y - m * x - c

    assert np.isclose(f(x0, y0), 0)
    assert np.isclose(f(x1, y1), 0)
    return f


def _equation_of_plane(a, b, c):
    """
    Deduce an expression for the equation of a plane passing through three points.

    Returns `None` if the points are colinear.

    :arg a: the first point the line passes through
    :type a: :class:`tuple`
    :arg b: the second point the line passes through
    :type b: :class:`tuple`
    :arg c: the third point the line passes through
    :type c: :class:`tuple`
    :returns: a function of three variables representing the plane
    :rtype: :class:`~.Callable`
    """
    plane = sympy.Plane(sympy.Point3D(a), sympy.Point3D(b), sympy.Point3D(c))

    def equation(x, y, z):
        return plane.distance(sympy.Point3D((x, y, z)))

    return equation


def equation_of_plane(*points):
    r"""
    Deduce an expression for the equation of a plane passing through a set of points.

    :arg points: the first point the line passes through
    :type points: :class:`tuple` of :class:`tuple`\s
    :returns: a function of three variables representing the plane
    :rtype: :class:`~.Callable`
    """
    assert len(points) >= 3
    indices = list(range(len(points)))
    while len(indices) >= 3:
        np.random.shuffle(indices)
        i, j, k = indices[:3]
        try:
            return _equation_of_plane(points[i], points[j], points[k])
        except ValueError:
            indices.pop(0)
    raise ValueError("Could not determine a plane for the provided points.")
