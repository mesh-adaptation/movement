import unittest

import numpy as np
from firedrake import *

from movement import LaplacianSmoother, SpringMover


class BaseClasses:
    """
    Base classes for testing mesh movement under forcings.
    """

    class TestBoundaryForcing(unittest.TestCase):
        """
        Unit tests for mesh movement under boundary forcings.
        """

        def move(
            self,
            mesh,
            fixed_boundary_tags=None,
            moving_boundary_tags=None,
            vector=None,
            **kwargs,
        ):
            vector = vector or [1, 0]
            mover = self.mover(mesh)
            bcs = []
            if fixed_boundary_tags:
                bcs.append(DirichletBC(mover.coord_space, 0, fixed_boundary_tags))
            if moving_boundary_tags:
                f = Function(mover.coord_space).interpolate(as_vector(vector))
                bcs.append(DirichletBC(mover.coord_space, f, moving_boundary_tags))
            mover.move(0.0, boundary_conditions=bcs, **kwargs)
            return mover.mesh

        @staticmethod
        def shifted_mesh(nx, ny, shift_x=0, shift_y=0):
            mesh = UnitSquareMesh(nx, ny)
            x, y = SpatialCoordinate(mesh)
            shifted_coords = Function(mesh.coordinates)
            shifted_coords.interpolate(as_vector([x + shift_x, y + shift_y]))
            return Mesh(shifted_coords)

        @staticmethod
        def stretched_mesh(nx, ny, stretch_x=1, stretch_y=1):
            return RectangleMesh(nx, ny, stretch_x, stretch_y)

        def test_fixed_triangle(self):
            mesh = UnitTriangleMesh()
            coords = mesh.coordinates.dat.data
            self.assertTrue(np.allclose(coords, self.move(mesh).coordinates.dat.data))

        def test_fixed_square(self):
            mesh = UnitSquareMesh(1, 1)
            coords = mesh.coordinates.dat.data
            self.assertTrue(np.allclose(coords, self.move(mesh).coordinates.dat.data))

        def test_force_right_free(self):
            mesh = UnitSquareMesh(10, 10)
            coord_array = mesh.coordinates.dat.data
            new_coord_array = self.move(
                mesh, moving_boundary_tags=1
            ).coordinates.dat.data
            self.assertFalse(np.allclose(coord_array, new_coord_array))
            shifted_coord_array = self.shifted_mesh(
                10, 10, shift_x=1
            ).coordinates.dat.data
            self.assertTrue(np.allclose(shifted_coord_array, new_coord_array))

        def test_force_right_fixed(self):
            mesh = UnitSquareMesh(10, 10)
            coord_array = mesh.coordinates.dat.data
            new_mesh = self.move(
                mesh, moving_boundary_tags=1, fixed_boundary_tags=2, vector=[0.1, 0]
            )
            new_coord_array = new_mesh.coordinates.dat.data
            self.assertFalse(np.allclose(coord_array, new_coord_array))
            # # TODO: Implement no-slip BCs for segments 3 and 4 (#99)
            # stretched_mesh = self.stretched_mesh(10, 10, stretch_x=2)
            # stretched_coord_array = stretched_mesh.coordinates.dat.data
            # self.assertTrue(np.allclose(stretched_coord_array, new_coord_array))

        def test_force_right_left_free(self):
            mesh = UnitSquareMesh(10, 10)
            coord_array = mesh.coordinates.dat.data
            mesh = self.move(mesh, moving_boundary_tags=1, vector=[1, 0])
            self.assertFalse(np.allclose(coord_array, mesh.coordinates.dat.data))
            mesh = self.move(mesh, moving_boundary_tags=1, vector=[-1, 0])
            self.assertTrue(np.allclose(coord_array, mesh.coordinates.dat.data))


class TestSpringMover(BaseClasses.TestBoundaryForcing):
    """
    Unit tests for the lineal spring method under boundary forcings.
    """

    @staticmethod
    def mover(mesh):
        return SpringMover(mesh, 1.0)

    def test_update_boundary_displacement(self):
        mesh = UnitSquareMesh(10, 10)
        coord_array = mesh.coordinates.dat.data
        mover = self.mover(mesh)
        forcing = Function(mover.coord_space)
        bc = DirichletBC(mover.coord_space, forcing, 1)

        def update_bc(time):
            forcing.interpolate(as_vector([cos(pi * time / 2), sin(pi * time / 2)]))

        mover.move(0.0, boundary_conditions=bc, update_boundary_displacement=update_bc)
        self.assertFalse(np.allclose(coord_array, mover.mesh.coordinates.dat.data))
        mover.move(1.0, boundary_conditions=bc, update_boundary_displacement=update_bc)
        self.assertTrue(np.allclose(coord_array, mesh.coordinates.dat.data))


class TestLaplacianSmoother(BaseClasses.TestBoundaryForcing):
    """
    Unit tests for Laplacian smoothing under boundary forcings.
    """

    @staticmethod
    def mover(mesh):
        return LaplacianSmoother(mesh, 1.0)

    def test_convergence_error(self):
        with self.assertRaises(ConvergenceError) as cm:
            self.move(
                UnitSquareMesh(10, 10),
                moving_boundary_tags=1,
                fixed_boundary_tags=2,
                vector=[1.0, 0],
                solver_parameters={"ksp_type": "cg", "ksp_max_it": 0},
            )
        self.assertEqual(str(cm.exception), "Solver failed to converge.")

    def test_update_boundary_velocity(self):
        mesh = UnitSquareMesh(10, 10)
        coord_array = mesh.coordinates.dat.data
        mover = LaplacianSmoother(mesh, 1.0)
        forcing = Function(mover.coord_space)
        bc = DirichletBC(mover.coord_space, forcing, 1)

        def update_bc(time):
            forcing.interpolate(as_vector([cos(pi * time / 2), sin(pi * time / 2)]))

        mover.move(0.0, boundary_conditions=bc, update_boundary_velocity=update_bc)
        self.assertFalse(np.allclose(coord_array, mover.mesh.coordinates.dat.data))
        mover.move(1.0, boundary_conditions=bc, update_boundary_velocity=update_bc)
        self.assertTrue(np.allclose(coord_array, mesh.coordinates.dat.data))
