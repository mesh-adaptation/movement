from firedrake import *
from movement import monge_ampere
import pytest


def monitor_function(mesh):
    return Constant(1.0)


@pytest.fixture(params=['relaxation', 'quasi_newton'])
def method(request):
    return request.param


def test_uniform_monitor(method):
    mesh = UnitSquareMesh(10, 10)
    coords = mesh.coordinates.dat.data.copy()
    num_iterations = monge_ampere(mesh, monitor_function, method=method)
    assert np.allclose(coords, mesh.coordinates.dat.data)
    assert num_iterations == 1
