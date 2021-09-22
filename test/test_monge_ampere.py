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
    assert num_iterations == 1


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

    assert num_it_continue < num_it_naive
    assert num_it_init + num_it_continue < num_it_naive


if __name__ == "__main__":
    test_continue('relaxation', exports=True)
    # test_continue('quasi_newton', exports=True)
