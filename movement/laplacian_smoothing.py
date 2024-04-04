import firedrake
from firedrake.petsc import PETSc
import ufl
import movement.solver_parameters as solver_parameters
from movement.mover import PrimeMover
import numpy as np


__all__ = ["LaplacianSmoother"]


class LaplacianSmoother(PrimeMover):
    r"""
    Movement of a ``mesh`` is driven by a mesh velocity :math:`\mathbf{v}`, which is
    determined by solving  a vector Poisson problem

    .. math::
        \nabla^2_{\boldsymbol{\xi}}\mathbf{v} = \mathbf{f}, \quad \boldsymbol{\xi}\in\Omega,

    with a forcing term :math:`\mathbf{f}` under Dirichlet boundary conditions

    .. math::
        \mathbf{v} = \mathbf{v}_D, \quad \boldsymbol{\xi}\in\partial\Omega,

    where the computational coordinates :math:`\boldsymbol{\xi} := \mathbf{x}(t_0)` are
    the physical coordinates at the beginning of the simulation.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, mesh, timestep, **kwargs):
        """
        :arg mesh: the physical mesh to be moved
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg timestep: the timestep length used
        :type timestep: :class:`float`
        """
        super().__init__(mesh, **kwargs)
        assert timestep > 0.0
        self.dt = timestep
        self.f = firedrake.Function(self.coord_space)
        dim = self.mesh.topological_dimension()
        self.displacement = np.zeros((self.mesh.num_vertices(), dim))

    @PETSc.Log.EventDecorator()
    def _setup_solver(self, boundary_conditions):
        if not hasattr(self, "_solver"):
            test = firedrake.TestFunction(self.coord_space)
            trial = firedrake.TrialFunction(self.coord_space)
            a = ufl.inner(ufl.grad(trial), ufl.grad(test)) * self.dx
            L = ufl.inner(self.f, test) * self.dx
            problem = firedrake.LinearVariationalProblem(
                a, L, self.v, bcs=boundary_conditions
            )
            self._solver = firedrake.LinearVariationalSolver(
                problem,
                solver_parameters=solver_parameters.cg,
            )
        self._solver.solve()

    @PETSc.Log.EventDecorator()
    def move(self, time, update_forcings=None, boundary_conditions=None):
        """
        Assemble and solve the Laplacian system and update the coordinates.

        :arg time: the current time
        :type time: :class:`float`
        :kwarg update_forcings: function that updates the forcing :attr:`f` and/or
            boundary conditions at the current time
        :type update_forcings: :class:`~.Callable` with a single argument of
            :class:`float` type
        :kwarg boundary_conditions: Dirichlet boundary conditions to be enforced
        :type boundary_conditions: :class:`~.DirichletBC` or :class:`list` thereof
        """
        if update_forcings is not None:
            update_forcings(time)
        if not boundary_conditions:
            boundary_conditions = firedrake.DirichletBC(
                self.coord_space, 0, "on_boundary"
            )
        self._setup_solver(boundary_conditions)

        # Solve on computational mesh
        self.mesh.coordinates.assign(self.xi)
        self._solver.solve()

        # Update mesh coordinates
        self.displacement[:] = self.v.dat.data_with_halos * self.dt
        self._x.dat.data_with_halos[:] += self.displacement
        self.mesh.coordinates.assign(self._x)
