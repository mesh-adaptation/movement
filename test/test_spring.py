import firedrake
from movement import SpringMover
import numpy as np
import os
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
    mesh = firedrake.UnitTriangleMesh()
    mover = SpringMover(mesh)
    facet_areas = mover.facet_areas.dat.data
    expected = [1, np.sqrt(2), 1]
    assert np.allclose(facet_areas, expected)


def test_tangents():
    """
    Test that the tangent vectors of a right-angled
    triangle are computed correctly.
    """
    mesh = firedrake.UnitTriangleMesh()
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
    mesh = firedrake.UnitTriangleMesh()
    mover = SpringMover(mesh)
    angles = 180*mover.angles.dat.data/np.pi
    expected = [0, 135, 90]
    assert np.allclose(angles, expected)


def test_forced(method, time, plot=False)
    """
    Test that a uniform mesh is moved as expected
    when only one of its boundaries is forced.
    """
    it = 0
    if not np.isclose(time, 0.0):
        it = 1/time
        assert np.isclose(np.round(it), it), "The time needs to be a reciprocal of an integer"
        it = int(np.round(it))

    # Set parameters
    n = 10   # Mesh resolution
    A = 0.5  # Forcing amplitude
    T = 1.0  # Forcing period

    # Construct mesh and mover
    mesh = firedrake.SquareMesh(n, n, 2)
    V = mesh.coordinates.function_space()
    coords = mesh.coordinates.dat.data.copy()
    mover = SpringMover(mesh, method=method)

    def update_forcings(t):
        """
        Sinusoidal forcing on the top boundary.
        """
        for i in firedrake.DirichletBC(V, 0, 4).nodes:
            mover.f.dat.data[i][1] += A*np.sin(2*np.pi*t/T)*np.sin(np.pi*coords[i][0])

    # Move the mesh
    mover.move(time, update_forcings=update_forcings, fixed_boundaries=[1, 2, 3])
    new_coords = mover.mesh.coordinates.dat.data

    # Plotting
    if plot:
        import matplotlib.pyplot as plt
        from firedrake.pyplot import triplot

        fig, axes = plt.subplots()
        triplot(mover.mesh, axes=axes)
        axes.axis(False)
        plt.tight_layout()
        plt.savefig(f"plots/mesh_{method}_{it}.png")

    # Check as expected
    pwd = os.path.dirname(__file__)
    fname = os.path.join(pwd, "data", f"forced_mesh_lineal_{it}.npy")
    expected = coords if np.isclose(time, 0.0) else np.load(fname)
    assert np.allclose(new_coords, expected)

    # Test that the Firedrake mesh coordinates and the underlying DMPlex's
    # coordinates are consistent after mesh movement.
    plex_coords = mover.plex.getCoordinatesLocal().array
    mesh_coords = mover.mesh.coordinates.dat.data_with_halos
    assert np.allclose(plex_coords.reshape(*mesh_coords.shape), mesh_coords)


if __name__ == "__main__":
    mesh = test_forced('lineal', 0.25, plot=True, test=False).mesh
    np.save("data/forced_mesh_lineal_4", mesh.coordinates.dat.data)
