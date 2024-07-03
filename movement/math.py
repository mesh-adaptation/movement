import numpy as np
import sympy

__all__ = []


def equation_of_hyperplane(*points):
    r"""
    Deduce an expression for the equation of a hyperplane passing through a set of
    points.

    :arg points: points the hyperplane passes through
    :type points: :class:`tuple` of :class:`tuple`\s
    :returns: a function representing the hyperplane
    :rtype: :class:`~.Callable`
    """
    dim = len(points[0])
    assert len(points) >= dim
    for point in points:
        assert len(point) == dim
    indices = list(range(len(points)))
    try:
        Point, Hyperplane, name = {
            2: (sympy.Point2D, sympy.Line, "line"),
            3: (sympy.Point3D, sympy.Plane, "plane"),
        }[dim]
    except KeyError as exc:
        raise NotImplementedError(
            f"equation_of_hyperplane not implemented in {dim}D."
        ) from exc
    while len(indices) >= dim:
        np.random.shuffle(indices)
        try:
            hyperplane = Hyperplane(*(Point(points[i]) for i in indices[:dim]))

            def equation(*xyz):
                return hyperplane.distance(Point(xyz))

            return equation
        except ValueError:
            indices.pop(0)
    raise ValueError(f"Could not determine a {name} for the provided points.")
