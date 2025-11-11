import abc
from warnings import warn

import firedrake
import firedrake.exceptions as fexc
import numpy as np
import ufl
from animate.utility import function_data_max, function_data_min, function_data_sum
from firedrake.cython.dmcommon import create_section
from firedrake.petsc import PETSc

from movement.tangling import MeshTanglingChecker

__all__ = ["PrimeMover"]


class PrimeMover(abc.ABC):
    """
    Base class for all mesh movers.
    """

    def __init__(
        self,
        mesh,
        monitor_function=None,
        raise_convergence_errors=True,
        tangling_check=True,
        quadrature_degree=None,
    ):
        r"""
        :arg mesh: the physical mesh
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg monitor_function: a Python function which takes a mesh as input
        :type monitor_function: :class:`~.Callable`
        :kwarg raise_convergence_errors: convergence error handling behaviour: if `True`
            then :class:`~.ConvergenceError`\s are raised, else warnings are raised and
            the program is allowed to continue
        :type raise_convergence_errors: :class:`bool`
        :kwarg tangling_check: check whether the mesh has tangled elements (on by
            default)
        :type tangling_check: :class:`bool`
        :kwarg quadrature_degree: quadrature degree to be passed to Firedrakes measures
        :type quadrature_degree: :class:`int`
        """
        self.mesh = firedrake.Mesh(mesh.coordinates.copy(deepcopy=True))
        self.monitor_function = monitor_function
        if not raise_convergence_errors:
            warn(
                f"{type(self)}.move called with raise_convergence_errors=False."
                " Beware: this option can produce poor quality meshes!",
                stacklevel=1,
            )
        self.raise_convergence_errors = raise_convergence_errors
        self.dim = self.mesh.topological_dimension
        self.gdim = self.mesh.geometric_dimension

        # DMPlex setup
        self.plex = self.mesh.topology_dm
        self.vertex_indices = self.plex.getDepthStratum(0)
        self.edge_indices = self.plex.getDepthStratum(1)
        entity_dofs = np.zeros(self.dim + 1, dtype=np.int32)
        entity_dofs[0] = self.gdim
        self._coordinate_section = create_section(self.mesh, entity_dofs)[0]
        dm_coords = self.plex.getCoordinateDM()
        dm_coords.setDefaultSection(self._coordinate_section)
        try:
            self._local_coordinates_vec = dm_coords.createLocalVec()
            self._update_plex_coordinates()
        except ValueError:
            warn("Cannot update DMPlex coordinates for periodic meshes.", stacklevel=1)
            self._local_coordinates_vec = None

        self.dx = firedrake.dx(domain=self.mesh, degree=quadrature_degree)
        self.ds = firedrake.ds(domain=self.mesh, degree=quadrature_degree)
        self.dS = firedrake.dS(domain=self.mesh, degree=quadrature_degree)

        self._create_function_spaces()
        self._create_functions()
        self._all_boundary_segments = self.mesh.exterior_facets.unique_markers

        # Utilities
        if tangling_check:
            self.tangling_checker = MeshTanglingChecker(
                self.mesh, raise_error=raise_convergence_errors
            )

    def _create_function_spaces(self):
        self.coord_space = self.mesh.coordinates.function_space()
        self.P0 = firedrake.FunctionSpace(self.mesh, "DG", 0)

    def _create_functions(self):
        self.x = firedrake.Function(self.mesh.coordinates, name="Physical coordinates")
        self.xi = firedrake.Function(
            self.mesh.coordinates, name="Computational coordinates"
        )
        self.v = firedrake.Function(self.coord_space, name="Mesh velocity")
        self.volume = firedrake.Function(self.P0, name="Mesh volume")
        self.volume.interpolate(ufl.CellVolume(self.mesh))

    def _convergence_message(self, iterations=None):
        """
        Report solver convergence.

        :kwarg iterations: number of iterations before reaching convergence
        :type iterations: :class:`int`
        """
        msg = "Solver converged"
        if iterations:
            msg += f" in {iterations} iteration{plural(iterations)}"
        PETSc.Sys.Print(f"{msg}.")

    def _exception(self, msg, exception=None, error_type=fexc.ConvergenceError):
        """
        Raise an error or warning as indicated by the :attr:`raise_convergence_error`
        option.

        :arg msg: message for the error/warning report
        :type msg: :class:`str`
        :kwarg exception: original exception that was triggered
        :type exception: :class:`~.Exception` object
        :kwarg error_type: error class to use
        :type error_type: :class:`~.Exception` class
        """
        exc_type = error_type if self.raise_convergence_errors else Warning
        if exception:
            raise exc_type(msg) from exception
        else:
            raise exc_type(msg)

    def _convergence_error(self, iterations=None, exception=None):
        """
        Raise an error or warning for a solver fail as indicated by the
        :attr:`raise_convergence_error` option.

        :kwarg iterations: number of iterations before failure
        :type iterations: :class:`int`
        :kwarg exception: original exception that was triggered
        :type exception: :class:`~.Exception`
        """
        msg = "Solver failed to converge"
        if iterations:
            msg += f" in {iterations} iteration{plural(iterations)}"
        self._exception(f"{msg}.", exception=exception)

    def _divergence_error(self, iterations=None, exception=None):
        """
        Raise an error or warning for a solver divergence as indicated by the
        :attr:`raise_convergence_error` option.

        :kwarg iterations: number of iterations before failure
        :type iterations: :class:`int`
        :kwarg exception: original exception that was triggered
        :type exception: :class:`~.Exception`
        """
        msg = "Solver diverged"
        if iterations:
            msg += f" after {iterations} iteration{plural(iterations)}"
        self._exception(f"{msg}.", exception=exception)

    def _update_plex_coordinates(self):
        """
        Update the underlying DMPlex coordinates with the coordinates of the Firedrake
        mesh.
        """
        if self._local_coordinates_vec is None:
            raise ValueError("Cannot update DMPlex coordinates for periodic meshes.")
        self._local_coordinates_vec.array[:] = np.reshape(
            self.mesh.coordinates.dat.data_with_halos,
            self._local_coordinates_vec.array.shape,
        )
        self.plex.setCoordinatesLocal(self._local_coordinates_vec)

    def _coordinate_offset(self, index):
        """
        Map the index of a DMPlex coordinate to the coordinate index in Firedrake.

        :arg index: DMPlex coordinate index
        :type index: :class:`int`
        """
        return self._coordinate_section.getOffset(index) // self.dim

    def _edge_offset(self, index):
        """
        Map the index of a DMPlex edge to the edge index in Firedrake.

        :arg index: DMPlex edge index
        :type index: :class:`int`
        """
        if not hasattr(self, "_edge_vector_section"):
            entity_dofs = np.zeros(self.dim + 1, dtype=np.int32)
            entity_dofs[1] = 1
            self._edge_vector_section = create_section(self.mesh, entity_dofs)[0]
        return self._edge_vector_section.getOffset(index)

    @property
    def volume_ratio(self):
        """
        :return: the ratio of the largest and smallest element volumes.
        :rtype: :class:`float`
        """
        return function_data_max(self.volume) / function_data_min(self.volume)

    @property
    def coefficient_of_variation(self):
        """
        :return: the coefficient of variation (σ/μ) of element volumes.
        :rtype: :class:`float`
        """
        size = self.volume.dat.dataset.layout_vec.getSizes()
        mean = function_data_sum(self.volume) / size
        coef = firedrake.Function(self.P0)
        coef.interpolate((self.volume - mean) ** 2)
        return np.sqrt(function_data_sum(coef) / size) / mean

    @abc.abstractmethod
    def move(self):
        """
        Move the mesh according to the method of choice.
        """
        pass  # pragma: no cover

    def to_physical_coordinates(self):
        r"""
        Switch coordinates to correspond to the physical mesh :class:`\mathcal{H}_P`.
        """
        self.mesh.coordinates.assign(self.x)

    def to_computational_coordinates(self):
        r"""
        Switch coordinates to correspond to the computational mesh
        :class:`\mathcal{H}_C`.
        """
        self.mesh.coordinates.assign(self.xi)


def plural(iterations):
    """
    :return: 's' if `iterations` should be referred to in the plural sense
    :rtype: :class:`str`
    """
    return "s" if iterations != 1 else ""
