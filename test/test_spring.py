import os

import firedrake
import numpy as np
import pytest

from movement import SpringMover


@pytest.fixture(params=["lineal"])
def method(request):
    return request.param


@pytest.fixture(params=[0.0, 0.25])
def time(request):
    return request.param


def test_facet_areas():
    """
    Test that the edge lenths of a right-angled
    triangle are computed correctly.
    """
    mesh = firedrake.UnitTriangleMesh()
    timestep = 1.0
    mover = SpringMover(mesh, timestep)
    facet_areas = mover.facet_areas.dat.data
    expected = [1, np.sqrt(2), 1]
    assert np.allclose(facet_areas, expected)


def test_tangents():
    """
    Test that the tangent vectors of a right-angled
    triangle are computed correctly.
    """
    mesh = firedrake.UnitTriangleMesh()
    timestep = 1.0
    mover = SpringMover(mesh, timestep)
    tangents = mover.tangents.dat.data
    expected = [[1, 0], [np.sqrt(2) / 2, np.sqrt(2) / 2], [0, 1]]
    assert np.allclose(np.abs(tangents), expected)


def test_angles():
    """
    Test that the arguments (angles from the
    x-axis) of a right-angled triangle are
    computed correctly.
    """
    mesh = firedrake.UnitTriangleMesh()
    timestep = 1.0
    mover = SpringMover(mesh, timestep)
    angles = 180 * mover.angles.dat.data / np.pi
    expected = [0, 135, 90]
    assert np.allclose(angles, expected)
