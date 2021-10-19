import firedrake
from firedrake.petsc import PETSc
import ufl
import numpy as np
import movement.solver_parameters as solver_parameters
from movement.mover import PrimeMover


__all__ = ["SpringMover_Lineal", "SpringMover_Torsional", "SpringMover"]


def SpringMover(mesh, method='lineal', **kwargs):
    """
    Movement of a ``mesh`` is determined by reinterpreting
    it as a structure of stiff beams and solving an
    associated discrete linear elasticity problem.

    See Farhat, Degand, Koobus and Lesoinne, "Torsional
    springs for two-dimensional dynamic unstructured fluid
    meshes" (1998), Computer methods in applied mechanics
    and engineering, 163:231-245.
    """
    if method == 'lineal':
        return SpringMover_Lineal(mesh, **kwargs)
    elif method == 'torsional':
        return SpringMover_Torsional(mesh, **kwargs)
    else:
        raise ValueError(f"Method {method} not recognised.")


class SpringMover_Base(PrimeMover):
    """
    Base class for mesh movers based on spring
    analogies.
    """
    @PETSc.Log.EventDecorator("SpringMover_Base.__init__")
    def __init__(self, mesh, **kwargs):
        """
        :arg mesh: the physical mesh
        """
        super().__init__(mesh)
        self.HDivTrace = firedrake.FunctionSpace(mesh, "HDiv Trace", 0)
        self.HDivTrace_vec = firedrake.VectorFunctionSpace(mesh, "HDiv Trace", 0)
        self.f = firedrake.Function(self.mesh.coordinates.function_space())
        self.displacement = np.zeros(mesh.num_vertices())

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

    @property
    @PETSc.Log.EventDecorator("SpringMover_Base.tangents")
    def tangents(self):
        """
        Compute tangent vectors for all edges in
        the mesh.
        """
        if not hasattr(self, '_tangents_solver'):
            test = firedrake.TestFunction(self.HDivTrace_vec)
            trial = firedrake.TrialFunction(self.HDivTrace_vec)
            self._tangents = firedrake.Function(self.HDivTrace_vec)
            n = ufl.FacetNormal(self.mesh)
            s = ufl.perp(n)
            a = ufl.inner(trial('+'), test('+'))*self.dS + ufl.inner(trial, test)*self.ds
            L = ufl.inner(test('+'), s('+'))*self.dS + ufl.inner(test, s)*self.ds
            prob = firedrake.LinearVariationalProblem(a, L, self._tangents)
            self._tangents_solver = firedrake.LinearVariationalSolver(
                prob, solver_parameters=solver_parameters.jacobi,
            )
        self._tangents_solver.solve()
        return self._tangents

    @property
    @PETSc.Log.EventDecorator("SpringMover_Base.angles")
    def angles(self):
        r"""
        Compute the argument of each edge in the
        mesh, i.e. its angle from the :math:`x`-axis
        in the :math:`x-y` plane.
        """
        t = self.tangents
        if not hasattr(self, '_angles_solver'):
            test = firedrake.TestFunction(self.HDivTrace)
            trial = firedrake.TrialFunction(self.HDivTrace)
            self._angles = firedrake.Function(self.HDivTrace)
            e0 = np.zeros(self.dim)
            e0[0] = 1.0
            X = ufl.as_vector(e0)
            a = trial('+')*test('+')*self.dS + trial*test*self.ds
            L = test('+')*ufl.dot(t('+'), X('+'))*self.dS + test*ufl.dot(t, X)*self.ds
            prob = firedrake.LinearVariationalProblem(a, L, self._angles)
            self._angles_solver = firedrake.LinearVariationalSolver(
                prob, solver_parameters=solver_parameters.jacobi,
            )
        self._angles_solver.solve()
        self._angles.dat.data_with_halos[:] = np.arccos(self._angles.dat.data_with_halos)
        return self._angles

    @property
    @PETSc.Log.EventDecorator("SpringMover_Base.stiffness_matrix")
    def stiffness_matrix(self):
        angles = self.angles
        edge_lengths = self.facet_areas
        bnd = self.mesh.exterior_facets
        N = self.mesh.num_vertices()

        K = np.zeros((2*N, 2*N))
        for e in range(*self.edge_indices):
            off = self.edge_vector_offset(e)
            i, j = (self.coordinate_offset(v) for v in self.plex.getCone(e))
            if bnd.point2facetnumber[e] != -1:
                K[2*i][2*i] += 1.0
                K[2*i+1][2*i+1] += 1.0
                K[2*j][2*j] += 1.0
                K[2*j+1][2*j+1] += 1.0
            else:
                l = edge_lengths.dat.data_with_halos[off]
                angle = angles.dat.data_with_halos[off]
                c = np.cos(angle)
                s = np.sin(angle)
                K[2*i][2*i] += c*c/l
                K[2*i][2*i+1] += s*c/l
                K[2*i][2*j] += -c*c/l
                K[2*i][2*j+1] += -s*c/l
                K[2*i+1][2*i] += s*c/l
                K[2*i+1][2*i+1] += s*s/l
                K[2*i+1][2*j] += -s*c/l
                K[2*i+1][2*j+1] += -s*s/l
                K[2*j][2*i] += -c*c/l
                K[2*j][2*i+1] += -s*c/l
                K[2*j][2*j] += c*c/l
                K[2*j][2*j+1] += s*c/l
                K[2*j+1][2*i] += -s*c/l
                K[2*j+1][2*i+1] += -s*s/l
                K[2*j+1][2*j] += s*c/l
                K[2*j+1][2*j+1] += s*s/l
        return K

    @PETSc.Log.EventDecorator("SpringMover_Lineal.apply_dirichlet_conditions")
    def apply_dirichlet_conditions(self, tags):
        """
        Enforce that nodes on certain tagged boundaries
        do not move.

        :arg tags: a list of boundary tags
        """
        bnd = self.mesh.exterior_facets
        if not set(tags).issubset(set(bnd.unique_markers)):
            raise ValueError(f"{tags} contains invalid boundary tags")
        subsets = sum([list(bnd.subset(physID).indices) for physID in tags], start=[])
        for e in range(*self.edge_indices):
            i, j = (self.coordinate_offset(v) for v in self.plex.getCone(e))
            if bnd.point2facetnumber[e] in subsets:
                self.displacement[2*i] = 0.0
                self.displacement[2*i+1] = 0.0
                self.displacement[2*j] = 0.0
                self.displacement[2*j+1] = 0.0


class SpringMover_Lineal(SpringMover_Base):
    """
    Movement of a ``mesh`` is determined by reinterpreting
    it as a structure of stiff beams and solving an
    associated discrete linear elasticity problem.

    We consider the 'lineal' case, as described in
    Farhat, Degand, Koobus and Lesoinne, "Torsional
    springs for two-dimensional dynamic unstructured fluid
    meshes" (1998), Computer methods in applied mechanics
    and engineering, 163:231-245.
    """
    @PETSc.Log.EventDecorator("SpringMover_Lineal.move")
    def move(self, time, update_forcings=None, fixed_boundaries=[]):
        """
        Assemble and solve the lineal spring system and
        update the coordinates.

        :arg time: the current time
        :kwarg update_forcings: function that updates
            the forcing :attr:`f` at the current time
        :kwarg fixed_boundaries: list of boundaries
            where Dirichlet conditions are to be
            enforced
        """
        if update_forcings is not None:
            update_forcings(time)

        # Assemble
        K = self.stiffness_matrix
        rhs = self.f.dat.data_with_halos.flatten()

        # Solve
        self.displacement = np.linalg.solve(K, rhs)

        # Enforce Dirichlet conditions as a post-process
        if len(fixed_boundaries) > 0:
            self.apply_dirichlet_conditions(fixed_boundaries)

        # Update mesh coordinates
        shape = self.mesh.coordinates.dat.data_with_halos.shape
        self.mesh.coordinates.dat.data_with_halos[:] += self.displacement.reshape(shape)
        self._update_plex_coordinates()


class SpringMover_Torsional(SpringMover_Lineal):
    """
    Movement of a ``mesh`` is determined by reinterpreting
    it as a structure of stiff beams and solving an
    associated discrete linear elasticity problem.

    We consider the 'torsional' case, as described in
    Farhat, Degand, Koobus and Lesoinne, "Torsional
    springs for two-dimensional dynamic unstructured fluid
    meshes" (1998), Computer methods in applied mechanics
    and engineering, 163:231-245.
    """
    @PETSc.Log.EventDecorator("SpringMover_Torsional.__init__")
    def __init__(self, mesh, **kwargs):
        """
        :arg mesh: the physical mesh
        """
        super().__init__(mesh, **kwargs)
        self.P0 = firedrake.FunctionSpace(mesh, "DG", 0)

    @property
    @PETSc.Log.EventDecorator("SpringMover_Torsional.areas")
    def areas(self):
        if not hasattr(self, '_areas'):
            self._areas = firedrake.Function(self.P0)
        self._areas.interpolate(ufl.CellVolume(self.mesh))
        return self._areas

    @property
    def K_lineal(self):
        return super(SpringMover_Torsional, self).stiffness_matrix

    @property
    @PETSc.Log.EventDecorator("SpringMover_Torsional.K_torsional")
    def K_torsional(self):
        tangents = self.tangents
        N = self.mesh.num_vertices()
        K = np.zeros((6*N, 6*N))

        # Get denominator
        denominator = self.areas.copy(deepcopy=True)
        denominator *= 4.0*denominator

        # Loop over all cells to construct the torsion matrix
        edge_lengths = self.facet_areas
        for c in range(*self.cell_indices):
            off = self.cell_offset(c)
            edges = self.plex.getCone(c)
            eij = edges.pop(0)
            i, j = self.plex.getCone(eij)
            k = [v for v in self.plex.getCone(edges[0]) if v not in (i, j)]
            assert len(k == 0)
            k = k[0]
            if i in self.plex.getCone(edges[0]):
                eki = edges.pop(0)
                ejk = edges.pop(1)
            else:
                eki = edges.pop(1)
                eik = edges.pop(1)
            assert len(edges) == 0, f"Cell {c} does not have three edges."

            # Get squared edge lengths
            lij = edge_lengths[self.edge_vector_offset(eij)]**2
            ljk = edge_lengths[self.edge_vector_offset(ejk)]**2
            lki = edge_lengths[self.edge_vector_offset(eki)]**2

            # Compute C
            C = np.diag(lki*lij, lij*ljk, ljk*lki)/denominator(off)

            # Compute R  # TODO: Check signs okay
            xij, yij = tangents.dat.data_with_halos[eij]
            xjk, yjk = tangents.dat.data_with_halos[ejk]
            xki, yki = tangents.dat.data_with_halos[eki]
            aij = xij/lij; aji = -aij; bij = yij/lij; bji = -bij
            ajk = xjk/ljk; akj = -ajk; bjk = yjk/ljk; bkj = -bjk
            aki = xki/lki; aik = -aki; bki = yki/lki; bik = -bki
            R = np.array(
                [[bik - bij, aij - aik, bij, -aij, -bik, aik],
                 [-bji, aji, bji - bjk, ajk - aji, bjk, -ajk],
                 [bki, -aki, -bkj, akj, bkj - bki, aki - akj]]
            )

            # Combine
            Kc = np.dot(np.dot(R.transpose(), C), R)

            # TODO: put in matrix

        return self._K_torsional

    @property
    @PETSc.Log.EventDecorator("SpringMover_Torsional.stiffness_matrix")
    def stiffness_matrix(self):
        raise NotImplementedError  # TODO: How to add contributions of different dim?
