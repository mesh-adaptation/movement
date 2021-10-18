import firedrake
from firedrake import PETSc
from pyadjoint import no_annotations
import ufl
from pyop2.profiling import timed_stage
import numpy as np
import movement.solver_parameters as solver_parameters
from movement.mover import PrimeMover


class SpringMover_Base(PrimeMover):
    """
    Base class for mesh movers based on spring
    analogies.
    """
    def __init__(self, mesh, **kwargs):
        """
        :arg mesh: the physical mesh
        """
        super().__init__(mesh)
        self.HDivTrace = firedrake.FunctionSpace(mesh, "HDiv Trace", 0)
        self.HDivTrace_vec = firedrake.VectorFunctionSpace(mesh, "HDiv Trace", 0)

    @property
    @PETSc.Log.EventDecorator("SpringMover_Base.facet_areas")
    def facet_areas(self):
        """
        Compute the areas of all facets in the
        mesh.

        In 2D, this corresponds to edge lengths.
        """
        if not hasattr(self, '_facet_area_solver'):
            test = firedrake.TestFunction(self.HDivTrace)
            trial = firedrake.TrialFunction(self.HDivTrace)
            self._facet_area = firedrake.Function(self.HDivTrace)
            A = ufl.FacetArea(self.mesh)
            a = trial('+')*test('+')*self.dS + trial*test*self.ds
            L = test('+')*A*self.dS + test*A*self.ds
            prob = firedrake.LinearVariationalProblem(a, L, self._facet_area)
            self._facet_area_solver = firedrake.LinearVariationalSolver(
                prob, solver_parameters=solver_parameters.jacobi,
            )
        self._facet_area_solver.solve()
        return self._facet_area
