import unittest
from unittest.mock import MagicMock

import numpy as np
from monitors import *
from parameterized import parameterized

from movement import *


class TestMongeAmpere(unittest.TestCase):
    """
    Unit tests for Monge-Ampere methods.
    """

    def mesh(self, dim, n=10):
        self.assertTrue(dim in (2, 3))
        return UnitSquareMesh(n, n) if dim == 2 else UnitCubeMesh(n, n, n)

    @property
    def dummy_mesh(self):
        return MagicMock(UnitSquareMesh)

    @property
    def dummy_monitor(self):
        return lambda *args: MagicMock(Function)

    def test_method_valueerror(self):
        with self.assertRaises(ValueError) as cm:
            MongeAmpereMover(self.dummy_mesh, self.dummy_monitor, method="method")
        self.assertEqual(str(cm.exception), "Method 'method' not recognised.")

    def test_no_monitor_valueerror(self):
        with self.assertRaises(ValueError) as cm:
            MongeAmpereMover(self.dummy_mesh, None)
        self.assertEqual(str(cm.exception), "Please supply a monitor function.")

    @parameterized.expand(
        [(2, "relaxation"), (2, "quasi_newton"), (3, "relaxation"), (3, "quasi_newton")]
    )
    def test_uniform_monitor(self, dim, method):
        """
        Test that the mesh mover converges in one iteration for a constant monitor
        function.
        """
        mesh = self.mesh(dim)
        coords = mesh.coordinates.dat.data.copy()
        P1 = FunctionSpace(mesh, "CG", 1)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)

        mover = MongeAmpereMover(
            mesh,
            const_monitor,
            method=method,
            phi_init=Function(P1),
            sigma_init=Function(P1_ten),
            rtol=1e-3,
        )
        num_iterations = mover.move()

        self.assertTrue(np.allclose(coords, mover.mesh.coordinates.dat.data))
        self.assertEqual(num_iterations, 0)

    @parameterized.expand(
        [(2, "relaxation"), (2, "quasi_newton"), (3, "relaxation"), (3, "quasi_newton")]
    )
    def test_continue(self, dim, method):
        """
        Test that providing a good initial guess benefits the solver.
        """
        mesh = self.mesh(dim)
        rtol = 1.0e-03

        # Solve the problem to a weak tolerance
        mover = MongeAmpereMover(mesh, ring_monitor, method=method, rtol=0.1)
        num_it_init = mover.move()

        # Continue with a tighter tolerance
        mover.rtol = rtol
        num_it_continue = mover.move()

        # Solve the problem again to a tight tolerance
        mesh = self.mesh(dim)
        mover = MongeAmpereMover(mesh, ring_monitor, method=method, rtol=rtol)
        num_it_naive = mover.move()

        self.assertLessEqual(num_it_continue, num_it_naive)
        self.assertLessEqual(num_it_init + num_it_continue, num_it_naive)
        # FIXME: Looks like the mesh is tangled or close to tangling
        #        for the relaxation method, which is concerning.

    @parameterized.expand(
        [(2, "relaxation"), (2, "quasi_newton"), (3, "relaxation"), (3, "quasi_newton")]
    )
    def test_change_monitor(self, dim, method):
        """
        Test that the mover can handle changes to the monitor function, such as would
        happen during timestepping.
        """
        mesh = self.mesh(dim)
        gdim = mesh.geometric_dimension()
        coords = mesh.coordinates.dat.data.copy()
        atol = 1.0e-03
        rtol = 1.0e02 * atol**gdim

        # Adapt to a ring monitor
        mover = MongeAmpereMover(mesh, ring_monitor, method=method, rtol=rtol)
        mover.move()
        moved_coords = mover.mesh.coordinates.dat.data
        self.assertFalse(np.allclose(coords, moved_coords, atol=atol))

        # Adapt to a constant monitor
        mover.monitor_function = const_monitor
        mover.move()
        moved_coords = mover.mesh.coordinates.dat.data
        self.assertTrue(np.allclose(coords, moved_coords, atol=atol))

    @parameterized.expand(
        [
            (2, "relaxation", True),
            (2, "relaxation", False),
            (2, "quasi_newton", True),
            (2, "quasi_newton", False),
            (3, "relaxation", True),
            (3, "relaxation", False),
            (3, "quasi_newton", True),
            (3, "quasi_newton", False),
        ]
    )
    def test_bcs(self, dim, method, fix_boundary):
        """
        Test that domain boundaries are fixed by the Monge-Ampere movers.
        """
        mesh = self.mesh(dim)
        bnd = assemble(Constant(1.0) * ds(domain=mesh))
        bnodes = DirichletBC(mesh.coordinates.function_space(), 0, "on_boundary").nodes
        bnd_coords = mesh.coordinates.dat.data.copy()[bnodes]

        # Adapt to a ring monitor
        mover = MongeAmpereMover(
            mesh,
            ring_monitor,
            method=method,
            fix_boundary_nodes=fix_boundary,
            rtol=1e-3,
        )
        mover.move()

        # Check boundary lengths are preserved
        bnd_new = assemble(Constant(1.0) * ds(domain=mover.mesh))
        self.assertAlmostEqual(bnd, bnd_new)

        # Check boundaries are indeed fixed
        if fix_boundary:
            bnd_coords_new = mover.mesh.coordinates.dat.data[bnodes]
            self.assertTrue(np.allclose(bnd_coords, bnd_coords_new))

    @parameterized.expand([("relaxation"), ("quasi_newton")])
    def test_maxiter_convergenceerror(self, method):
        """
        Test that the mesh mover raises a :class:`~.ConvergenceError` if it reaches the
        maximum number of iterations.
        """
        mesh = self.mesh(2, n=4)
        mover = MongeAmpereMover(mesh, ring_monitor, method=method, maxiter=1)
        with self.assertRaises(ConvergenceError) as cm:
            mover.move()
        self.assertEqual(str(cm.exception), "Failed to converge in 1 iteration.")

    def test_divergence_convergenceerror(self):
        """
        Test that the mesh mover raises a :class:`~.ConvergenceError` if it diverges.
        """
        mesh = self.mesh(2, n=4)
        mover = MongeAmpereMover_Relaxation(mesh, ring_monitor, dtol=1.0e-08)
        with self.assertRaises(ConvergenceError) as cm:
            mover.move()
        self.assertEqual(str(cm.exception), "Diverged after 1 iteration.")

    def test_initial_guess_valueerror(self):
        mesh = self.mesh(2, n=2)
        phi_init = Function(FunctionSpace(mesh, "CG", 1))
        with self.assertRaises(ValueError) as cm:
            MongeAmpereMover_Relaxation(mesh, ring_monitor, phi_init=phi_init)
        self.assertEqual(str(cm.exception), "Need to initialise both phi *and* sigma.")

    def test_coordinate_update(self):
        mesh = self.mesh(2, n=2)
        dg_coords = Function(VectorFunctionSpace(mesh, "DG", 1))
        dg_coords.project(mesh.coordinates)
        mover = MongeAmpereMover(Mesh(dg_coords), const_monitor)
        mover._grad_phi.assign(2.0)
        mover._update_coordinates()
        self.assertNotEqual(mover.grad_phi.dof_dset.size, mover._grad_phi.dof_dset.size)
        self.assertAlmostEqual(errornorm(mover.grad_phi, mover._grad_phi), 0)
        mover.xi += 2
        self.assertAlmostEqual(errornorm(mover.x, mover.xi), 0)


class TestMongeAmpereInterpolation(unittest.TestCase):
    """
    Unit tests for Monge-Ampere methods.
    """

    def mesh(self, dim, n=10):
        self.assertTrue(dim in (2, 3))
        return UnitSquareMesh(n, n) if dim == 2 else UnitCubeMesh(n, n, n)

    @parameterized.expand(
        [(2, "relaxation"), (2, "quasi_newton"), (3, "relaxation"), (3, "quasi_newton")]
    )
    def test_quadratic_transformation(self, dim, method):
        r"""Test for a known, quadratic movement :math:`x_i(\xi) = a_i \xi_i^2 + b_i \xi_i`
        where :math:`a_=(f_i-1)/(f_i+1)` and :math:`b=2/(f_i+1)`. This corresponds to a monitor
        of :math:`m(\xi)=\prod_i 1/((1-f_i)\xi_i + 1`, such that :math:`m(0)=1` and :math:`m(1)=1/f`,
        i.e. we get a quadratic variation in each direction, where the right most cell is a factor f_i
        larger than the left-most cell in each direction.

        We test for various interpolation methods to address issue #89."""
        n = 10
        f = np.arange(0, dim) + 2
        rtol = 1e-4

        mesh = self.mesh(dim, n=n)
        mover = MongeAmpereMover(mesh, reciprocal_monitor(f), method=method, rtol=1)
        # default tolerance is 0.5, which gives an overlap of 0.5 (in physical! coordinates) for each
        # bounding box around an element - meaning that for a unit square mesh the spatial index becomes
        # useless (with the default of 0.5) as every point in the domain overlaps with almost every single bounding box
        mover.mesh.tolerance = 1e-5
        x_to_test = [0.5] * dim
        vom = VertexOnlyMesh(mover.mesh, [x_to_test])
        P0DG_vom = VectorFunctionSpace(vom, "DG", 0)
        xtst = interpolate(mover.mesh.coordinates, P0DG_vom)
        self.assertAlmostEqual(np.linalg.norm(xtst.dat.data - x_to_test), 0)
        mover.move()
        xtst = interpolate(mover.mesh.coordinates, P0DG_vom)
        # self.assertAlmostEqual(np.linalg.norm(xtst.dat.data - x_to_test), 0)
        self.assertAlmostEqual(
            np.linalg.norm(mover.mesh.coordinates.at(x_to_test) - x_to_test), 0
        )
        mover.rtol = rtol
        # need to re-create var. solver for this to take effect:
        if method == "quasi_newton":
            del mover._equidistributor
        mover.move()
        xtst = interpolate(mover.mesh.coordinates, P0DG_vom)
        # TODO: this fails!
        # self.assertAlmostEqual(np.linalg.norm(xtst.dat.data - x_to_test), 0)
        self.assertAlmostEqual(
            np.linalg.norm(mover.mesh.coordinates.at(x_to_test) - x_to_test), 0
        )

        # test coordinates of moved nodes:
        Xorig = mesh.coordinates.dat.data[:]
        X = mover.mesh.coordinates.dat.data[:]
        # get ordering of nodes that is the same as np.meshgrid
        idx = np.lexsort(Xorig.T[::-1, :])
        xi = np.linspace(0, 1, n + 1)
        xi_grid = np.meshgrid(*([xi] * dim), indexing="ij")
        for i, fi in enumerate(f):
            a = (fi - 1) / (fi + 1)
            b = 2 / (fi + 1)
            x_xi = a * xi_grid[i] ** 2 + b * xi_grid[i]
            tol = 0.1 / n if dim == 2 else 0.15 / n
            self.assertLessEqual(
                np.abs(X[idx, i].reshape([n + 1] * dim) - x_xi).max(), tol
            )
