from firedrake import SquareMesh, UnitTriangleMesh
from movement import SpringMover
import numpy as np
import pytest


@pytest.fixture(params=['lineal'])
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
    mesh = UnitTriangleMesh()
    mover = SpringMover(mesh)
    facet_areas = mover.facet_areas.dat.data
    expected = [1, np.sqrt(2), 1]
    assert np.allclose(facet_areas, expected)


def test_tangents():
    """
    Test that the tangent vectors of a right-angled
    triangle are computed correctly.
    """
    mesh = UnitTriangleMesh()
    mover = SpringMover(mesh)
    tangents = mover.tangents.dat.data
    expected = [[1, 0], [np.sqrt(2)/2, np.sqrt(2)/2], [0, 1]]
    assert np.allclose(np.abs(tangents), expected)


def test_angles():
    """
    Test that the arguments (angles from the
    x-axis) of a right-angled triangle are
    computed correctly.
    """
    mesh = UnitTriangleMesh()
    mover = SpringMover(mesh)
    angles = 180*mover.angles.dat.data/np.pi
    expected = [0, 135, 90]
    assert np.allclose(angles, expected)


def test_forced(method, time):
    """
    Test that a uniform mesh is moved as expected
    when only one of its boundaries is forced.
    """

    # Set parameters
    n = 10   # Mesh resolution
    A = 0.5  # Forcing amplitude
    T = 1.0  # Forcing period

    # Construct mesh and mover
    mesh = SquareMesh(n, n, 2)
    coords = mesh.coordinates.dat.data.copy()
    bnd = mesh.exterior_facets
    mover = SpringMover(mesh, method)

    def update_forcings(t):
        """
        Sinusoidal forcing on the top boundary.
        """
        forced_nodes = list(bnd.subset(4).indices)
        for e in range(*mover.edge_indices):
            i, j = (mover.coordinate_offset(v) for v in mover.plex.getCone(e))
            facet_num = bnd.point2facetnumber[e]
            if facet_num in forced_nodes:
                mover.f.dat.data[i][1] += A*np.sin(2*np.pi*t/T)*np.sin(np.pi*coords[i][0])
                mover.f.dat.data[j][1] += A*np.sin(2*np.pi*t/T)*np.sin(np.pi*coords[j][0])

    # Move the mesh
    mover.move(time, update_forcings=update_forcings, fixed_boundaries=[1, 2, 3])
    new_coords = mover.mesh.coordinates.dat.data
    if np.isclose(time, 0.0):
        assert np.allclose(coords, new_coords)
    else:
        it = 1/time
        assert np.isclose(np.round(it), it), "The time needs to be a reciprocal of an integer"
        it = int(np.round(it))
        assert np.allclose(new_coords, np.load(f"data/forced_mesh_lineal_{it}.npy"))
    return mover.mesh


if __name__ == "__main__":
    it = 4
    mesh = test_forced('lineal', 1/it)
    np.save(f"data/forced_mesh_lineal_{it}", mesh.coordinates.dat.data)
