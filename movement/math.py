import numpy as np

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
