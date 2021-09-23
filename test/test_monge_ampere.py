from movement import *
from monitors import *
import pytest


@pytest.fixture(params=['relaxation', 'quasi_newton'])
def method(request):
    return request.param


def test_uniform_monitor(method, exports=False):
    """
    Test that the mesh mover converges in one
    iteration for a constant monitor function.
    """
    n = 10
    mesh = UnitSquareMesh(n, n)
    coords = mesh.coordinates.dat.data.copy()

    mover = MongeAmpereMover(mesh, const_monitor, method=method)
    num_iterations = mover.adapt()

    assert np.allclose(coords, mesh.coordinates.dat.data)
    assert num_iterations == 0


def test_continue(method, exports=False):
    """
    Test that providing a good initial guess
    benefits the solver.
    """
    n = 20
    mesh = UnitSquareMesh(n, n)
    rtol = 1.0e-03

    # Solve the problem to a weak tolerance
    mover = MongeAmpereMover(mesh, ring_monitor, method=method, rtol=0.1)
    num_it_init = mover.adapt()
    phi, sigma = mover.phi, mover.sigma
    if exports:
        File("outputs/init.pvd").write(phi, sigma)

    # Continue with a new Mover
    mover = MongeAmpereMover(mesh, ring_monitor, method=method, rtol=rtol,
                             phi_init=phi, sigma_init=sigma)
    num_it_continue = mover.adapt()
    if exports:
        File("outputs/continue.pvd").write(mover.phi, mover.sigma)

    # Solve the problem again to a tight tolerance
    mesh = UnitSquareMesh(n, n)
    mover = MongeAmpereMover(mesh, ring_monitor, method=method, rtol=rtol)
    num_it_naive = mover.adapt()
    if exports:
        File("outputs/naive.pvd").write(mover.phi, mover.sigma)

    assert num_it_continue <= num_it_naive
    assert num_it_init + num_it_continue <= num_it_naive
    # FIXME: Looks like the mesh is tangled or close to tangling
    #        for the relaxation method, which is concerning.


def test_change_monitor(method, exports=False):
    """
    Test that the mover can handle changes to
    the monitor function, such as would happen
    during timestepping.
    """
    n = 20
    mesh = UnitSquareMesh(n, n)
    coords = mesh.coordinates.dat.data.copy()
    tol = 1.0e-03

    # Adapt to a ring monitor
    mover = MongeAmpereMover(mesh, ring_monitor, method=method, rtol=tol)
    mover.adapt()
    if exports:
        File("outputs/ring.pvd").write(mover.phi, mover.sigma)
    assert not np.allclose(coords, mesh.coordinates.dat.data, atol=tol)

    # Adapt to a constant monitor
    mover.monitor_function = const_monitor
    mover.adapt()
    if exports:
        File("outputs/const.pvd").write(mover.phi, mover.sigma)
    assert np.allclose(coords, mesh.coordinates.dat.data, atol=tol)
