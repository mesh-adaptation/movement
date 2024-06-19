import unittest
from unittest.mock import MagicMock

import numpy as np
from firedrake import *

from movement import SpringMover


class TestExceptions(unittest.TestCase):
    """
    Unit tests for exceptions raised by spring-based Movers.
    """

    def setUp(self):
        self.mesh = UnitTriangleMesh()

    def test_method_valueerror(self):
        with self.assertRaises(ValueError) as cm:
            SpringMover(self.mesh, 1.0, method="method")
        self.assertEqual(str(cm.exception), "Method 'method' not recognised.")

    def test_torsional_notimplementederror(self):  # TODO: (#36)
        with self.assertRaises(NotImplementedError) as cm:
            SpringMover(self.mesh, 1.0, method="torsional")
        self.assertEqual(str(cm.exception), "Torsional springs not yet implemented.")

    def test_invalid_bc_space_valueerror(self):
        mover = SpringMover(self.mesh, 1.0)
        bc = DirichletBC(VectorFunctionSpace(self.mesh, "CG", 2), 0, "on_boundary")
        with self.assertRaises(ValueError) as cm:
            mover.assemble_stiffness_matrix(bc)
        msg = (
            "Boundary conditions must have SpringMover_Lineal.coord_space as their"
            " function space."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_invalid_bc_tag_valueerror(self):
        mover = SpringMover(self.mesh, 1.0)
        bc = DirichletBC(mover.coord_space, 0, 4)
        with self.assertRaises(ValueError) as cm:
            mover.assemble_stiffness_matrix(bc)
        msg = "[4] contains invalid boundary tags."
        self.assertEqual(str(cm.exception), msg)

    def test_convergence_error(self):
        mover = SpringMover(self.mesh, 1.0)
        N = 2 * self.mesh.num_vertices()
        mover.assemble_stiffness_matrix = MagicMock(return_value=np.zeros((N, N)))
        with self.assertRaises(ConvergenceError) as cm:
            mover.move(0)
        self.assertEqual(str(cm.exception), "Solver failed to converge.")


class TestStiffness(unittest.TestCase):
    """
    Unit tests for the stiffness matrices used by the spring-based Movers.
    """

    def test_boundary_conditions_triangle_all_boundary(self):
        """
        Test that applying boundary conditions over the whole boundary of a mesh with
        only one element gives an identity stiffness matrix.
        """
        mesh = UnitTriangleMesh()
        mover = SpringMover(mesh, 1.0, method="lineal")
        I = np.eye(2 * mesh.num_vertices())
        K = mover._stiffness_matrix()
        self.assertFalse(np.allclose(K, I))
        K_bc = mover.assemble_stiffness_matrix()
        self.assertFalse(np.allclose(K, K_bc))
        self.assertTrue(np.allclose(K_bc, I))

    def test_boundary_conditions_triangle_one_segment(self):
        """
        Test that applying boundary conditions over the whole boundary of a mesh with
        only one element modifies entries of the stiffness matrix to insert an identity
        matrix.
        """
        mesh = UnitTriangleMesh()
        mover = SpringMover(mesh, 1.0, method="lineal")
        K = mover._stiffness_matrix()
        I = np.eye(2 * mesh.num_vertices())
        self.assertFalse(np.allclose(K, I))
        bc = DirichletBC(mover.coord_space, 0, 1)
        K_bc = mover.assemble_stiffness_matrix(bc)
        self.assertFalse(np.allclose(K, K_bc))
        self.assertFalse(np.allclose(K_bc, I))
        self.assertTrue(np.allclose(np.where(np.isclose(K, K_bc), I, K_bc), I))

    def test_boundary_conditions_1x1_square_all_boundary(self):
        """
        Test that applying boundary conditions over the whole boundary of a mesh with
        only one non-boundary edge gives an identity stiffness matrix.
        """
        mesh = UnitSquareMesh(1, 1)
        mover = SpringMover(mesh, 1.0, method="lineal")
        K = mover._stiffness_matrix()
        I = np.eye(2 * mesh.num_vertices())
        self.assertFalse(np.allclose(K, I))
        K_bc = mover.assemble_stiffness_matrix()
        self.assertFalse(np.allclose(K, K_bc))
        self.assertTrue(np.allclose(K_bc, I))


class TestQuantities(unittest.TestCase):
    """
    Unit tests for the element quantity utilities underlying the :class:`~.SpringMover`
    classes.
    """

    def setUp(self):
        self.mover = SpringMover(UnitTriangleMesh(), 1.0)

    @staticmethod
    def rad2deg(radians):
        return 180 * radians / np.pi

    def test_facet_areas(self):
        """
        Test that the edge lenths of a right-angled triangle are computed correctly.
        """
        facet_areas = self.mover.facet_areas.dat.data
        self.assertTrue(np.allclose(facet_areas, [1, np.sqrt(2), 1]))

    def test_tangents(self):
        """
        Test that the tangent vectors of a right-angled triangle are computed correctly.
        """
        tangents = self.mover.tangents.dat.data
        expected = [[1, 0], [np.sqrt(2) / 2, np.sqrt(2) / 2], [0, 1]]
        self.assertTrue(np.allclose(np.abs(tangents), expected))

    def test_angles(self):
        """
        Test that the arguments (angles from the x-axis) of a right-angled triangle are
        computed correctly.
        """
        angles = self.rad2deg(self.mover.angles.dat.data)
        self.assertTrue(np.allclose(angles, [0, 135, 90]))


class TestMovement(unittest.TestCase):
    """
    Unit tests for movement under spring-based methods.
    """

    def test_fixed_triangle(self):
        mesh = UnitTriangleMesh()
        coords = mesh.coordinates.dat.data
        mover = SpringMover(mesh, 1.0)
        mover.move(0.0)
        self.assertTrue(np.allclose(coords, mover.mesh.coordinates.dat.data))

    def test_fixed_square(self):
        mesh = UnitSquareMesh(1, 1)
        coords = mesh.coordinates.dat.data
        mover = SpringMover(mesh, 1.0)
        mover.move(0.0)
        self.assertTrue(np.allclose(coords, mover.mesh.coordinates.dat.data))

    def test_force_right_free(self):
        mesh = UnitSquareMesh(10, 10)
        coord_array = mesh.coordinates.dat.data
        mover = SpringMover(mesh, 1.0)
        right = interpolate(as_vector([1, 0]), mover.coord_space)
        mover.move(0, boundary_conditions=DirichletBC(mover.coord_space, right, 2))
        new_coord_array = mover.mesh.coordinates.dat.data
        self.assertFalse(np.allclose(coord_array, new_coord_array))

        original_mesh = UnitSquareMesh(10, 10)
        x, y = SpatialCoordinate(original_mesh)
        shifted_coords = Function(original_mesh.coordinates)
        shifted_coords.interpolate(as_vector([x + 1, y]))
        shifted_mesh = Mesh(shifted_coords)
        shifted_coord_array = shifted_mesh.coordinates.dat.data
        self.assertTrue(np.allclose(shifted_coord_array, new_coord_array))

    def test_force_right_fixed(self):
        mesh = UnitSquareMesh(10, 10)
        coord_array = mesh.coordinates.dat.data
        mover = SpringMover(mesh, 1.0)
        right = interpolate(as_vector([1, 0]), mover.coord_space)
        moving_bc = DirichletBC(mover.coord_space, right, 2)
        fixed_bc = DirichletBC(mover.coord_space, 0, 1)
        mover.move(0, boundary_conditions=[fixed_bc, moving_bc])
        new_coord_array = mover.mesh.coordinates.dat.data
        self.assertFalse(np.allclose(coord_array, new_coord_array))

        # # TODO: Implement no-slip BCs for segments 3 and 4 (#99)
        # stretched_mesh = RectangleMesh(10, 10, 2, 1)
        # stretched_coord_array = stretched_mesh.coordinates.dat.data
        # plot_meshes(mover.mesh, stretched_mesh)
        # self.assertTrue(np.allclose(stretched_coord_array, new_coord_array))
