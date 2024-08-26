"""
Mesh movement based on solutions of equations of Monge-Ampère type.
"""

import abc
from collections.abc import Iterable

import firedrake
import firedrake.exceptions as fexc
import numpy as np
import ufl
from firedrake.petsc import PETSc
from pyadjoint import no_annotations

import movement.solver_parameters as solver_parameters
from movement.math import equation_of_hyperplane
from movement.mover import PrimeMover

__all__ = [
    "MongeAmpereMover_Relaxation",
    "MongeAmpereMover_QuasiNewton",
    "MongeAmpereMover",
]


def MongeAmpereMover(mesh, monitor_function, method="relaxation", **kwargs):
    r"""
    Factory function for generating Monge-Ampère mesh movers.

    Movement of a *mesh* is determined by a *monitor_function* :math:`m` and an equation
    of Monge-Ampère type,

    .. math::
        m(\mathbf{x})\det(\mathbf{I} + \mathbf{H}(\phi)) = \theta,

    for a convex scalar potential :math:`\phi=\phi(\boldsymbol{\xi})`, which is a
    function of the coordinates of the *computational* mesh. Here :math:`m=m(\mathbf{x})`
    is a function of the coordinates of the *physical* mesh, :math:`\mathbf{I}` is the
    identity matrix, :math:`\theta` is a normalisation coefficient  and
    :math:`\mathbf{H}(\phi)` denotes the Hessian of :math:`\phi` with respect to
    :math:`\boldsymbol{\xi}`.

    Different implementations solve the Monge-Ampère equation in different ways. If the
    `method` argument is set to `"relaxation"` then it is solved in parabolised form in
    :class:`~.MongeAmpereMover_Relaxation`. If the argument is set to `"quasi_newton"`
    then it is solved in its elliptic form using a quasi-Newton method in
    :class:`~.MongeAmpereMover_QuasiNewton`. Descriptions of both methods may be found in
    :cite:`MCB:18`.

    The physical mesh coordinates :math:`\mathbf{x}` are updated according to

    .. math::
        \mathbf{x} = \boldsymbol{\xi} + \nabla_{\boldsymbol{\xi}}\phi.

    :arg mesh: the physical mesh
    :type mesh: :class:`firedrake.mesh.MeshGeometry`
    :arg monitor_function: a Python function which takes a mesh as input
    :type monitor_function: :class:`~.Callable`
    :kwarg method: choose from 'relaxation' and 'quasi_newton'
    :type method: :class:`str`
    :kwarg phi_init: initial guess for the scalar potential
    :type phi_init: :class:`firedrake.function.Function`
    :kwarg H_init: initial guess for the Hessian
    :type H_init: :class:`firedrake.function.Function`
    :kwarg maxiter: maximum number of iterations for the solver
    :type maxiter: :class:`int`
    :kwarg rtol: relative tolerance for the residual
    :type rtol: :class:`float`
    :kwarg dtol: divergence tolerance for the residual
    :type dtol: :class:`float`
    :kwarg pseudo_timestep: pseudo-timestep (only relevant to relaxation method)
    :type pseudo_timestep: :class:`float`
    :kwarg fix_boundary_nodes: should all boundary nodes remain fixed?
    :type fix_boundary_nodes: :class:`bool`
    :return: the Monge-Ampere Mover object
    :rtype: :class:`MongeAmpereMover_Relaxation` or
        :class:`MongeAmpereMover_QuasiNewton`
    """
    implemented_methods = {
        "relaxation": MongeAmpereMover_Relaxation,
        "quasi_newton": MongeAmpereMover_QuasiNewton,
    }
    try:
        return implemented_methods[method](mesh, monitor_function, **kwargs)
    except KeyError:
        raise ValueError(f"Method '{method}' not recognised.")


def tangential(v, n):
    """
    Return component of `v` perpendicular to `n` (assumed normalised).

    This is used to project vectors onto the tangent plane of a boundary.

    :arg v: the vector to project
    :type v: :class:`ufl.Expr`
    :arg n: the normal vector
    :type n: :class:`ufl.Expr`
    """
    return v - ufl.dot(v, n) * n


class MongeAmpereMover_Base(PrimeMover, metaclass=abc.ABCMeta):
    """
    Base class for mesh movers based on the solution of Monge-Ampère type equations.

    Currently implemented subclasses: :class:`~.MongeAmpereMover_Relaxation` and
    :class:`~.MongeAmpereMover_QuasiNewton`. Descriptions of both methods may be found in
    :cite:`MCB:18`.
    """

    def __init__(self, mesh, monitor_function, **kwargs):
        """
        :arg mesh: the physical mesh
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg monitor_function: a Python function which takes a mesh as input
        :type monitor_function: :class:`~.Callable`
        :kwarg phi_init: initial guess for the scalar potential
        :type phi_init: :class:`firedrake.function.Function`
        :kwarg H_init: initial guess for the Hessian
        :type H_init: :class:`firedrake.function.Function`
        :kwarg maxiter: maximum number of iterations for the relaxation
        :type maxiter: :class:`int`
        :kwarg rtol: relative tolerance for the residual
        :type rtol: :class:`float`
        :kwarg dtol: divergence tolerance for the residual
        :type dtol: :class:`float`
        :kwarg fixed_boundary_segments: boundary segments to hold fixed
        :type fixed_boundary_segments: :class:`list` of :class:`str` or :class:`int`
        """
        if monitor_function is None:
            raise ValueError("Please supply a monitor function.")
        if mesh.coordinates.ufl_element().degree() != 1:
            raise NotImplementedError(
                f"{type(self).__name__} not implemented on curved meshes."
            )  # TODO: (#107)

        # Collect parameters before calling super
        self.maxiter = kwargs.pop("maxiter", 1000)
        self.rtol = kwargs.pop("rtol", 1.0e-08)
        self.dtol = kwargs.pop("dtol", 2.0)
        self.fixed_boundary_segments = kwargs.pop("fixed_boundary_segments", [])
        if self.fixed_boundary_segments == "on_boundary":
            self.fixed_boundary_segments = mesh.exterior_facets.unique_markers
        elif not isinstance(self.fixed_boundary_segments, Iterable):
            self.fixed_boundary_segments = [self.fixed_boundary_segments]
        super().__init__(mesh, monitor_function=monitor_function, **kwargs)
        self.theta = firedrake.Constant(0.0)

    def _create_function_spaces(self):
        super()._create_function_spaces()
        self.P1 = firedrake.FunctionSpace(self.mesh, "CG", 1)
        self.P1_vec = firedrake.VectorFunctionSpace(self.mesh, "CG", 1)
        self.P1_ten = firedrake.TensorFunctionSpace(self.mesh, "CG", 1)

    @abc.abstractmethod
    def _create_functions(self):
        super()._create_functions()
        self.monitor = firedrake.Function(self.P1, name="Monitor function")
        self._grad_phi = firedrake.Function(self.P1_vec)
        self.grad_phi = firedrake.Function(self.mesh.coordinates)
        self.monitor.interpolate(self.monitor_function(self.mesh))
        self.original_volume = firedrake.Function(self.volume)
        self.total_volume = firedrake.assemble(firedrake.Constant(1.0) * self.dx)
        self.L_P0 = firedrake.TestFunction(self.P0) * self.monitor * self.dx

    @PETSc.Log.EventDecorator()
    def apply_initial_guess(self, phi_init, H_init):
        """
        Initialise the approximations to the scalar potential and its Hessian with an
        initial guess.

        By default, both are initialised to zero, which corresponds to the case where the
        computational and physical meshes coincide.

        :arg phi_init: initial guess for the scalar potential
        :type phi_init: :class:`firedrake.function.Function`
        :arg H_init: initial guess for the Hessian
        :type H_init: :class:`firedrake.function.Function`
        """
        if phi_init is not None and H_init is not None:
            if self.dim == 1:
                self.phi.interpolate(phi_init)
                self.H.interpolate(H_init)
            else:
                self.phi.project(phi_init)
                self.H.project(H_init)
            self.phi_old.assign(self.phi)
            self.H_old.assign(self.H)
        elif phi_init is not None or H_init is not None:
            raise ValueError("Need to initialise both phi *and* H.")

    @property
    def relative_l2_residual(self):
        """
        :return: the relative :math:`L^2` norm residual.
        :rtype: :class:`float`
        """
        assert hasattr(self, "_residual_l2_form")
        assert hasattr(self, "_norm_l2_form")
        return (
            firedrake.assemble(self._residual_l2_form).dat.norm
            / firedrake.assemble(self._norm_l2_form).dat.norm
        )

    @PETSc.Log.EventDecorator()
    def _update_physical_coordinates(self):
        r"""
        Update the physical coordinates :math:`\mathbf{x}` using the recovered gradient:

        .. math::
            \mathbf{x} = \boldsymbol{\xi} + \nabla_{\boldsymbol{\xi}}\phi.

        After updating the coordinates, this method also checks for mesh tangling if this
        is turned on. (It will be turned on by default in the 2D case.)
        """
        try:
            self.grad_phi.assign(self._grad_phi)
        except Exception:
            self.grad_phi.interpolate(self._grad_phi)
        self.x.assign(self.xi + self.grad_phi)

        # Check if the mesh has become tangled
        if hasattr(self, "tangling_checker"):
            self.tangling_checker.check()

    def _l2_projector_bcs(self, boundary_tag):
        r"""
        Determine boundary conditions to apply for the :math:`L^2` projection step for
        a given boundary segment.

        Note that the boundary segment *must* be a straight line or plane.

        :arg boundary_tag: tag for the boundary segment in consideration
        :type boundary_tag: :class:`str`
        :returns: tuple of boundary conditions
        :rtype: :class:`tuple` of :class:`~.DirichletBC`\s
        """
        zero_bc = firedrake.DirichletBC(self.P1_vec, 0, boundary_tag)
        if boundary_tag in self.fixed_boundary_segments or self.dim == 1:
            return (zero_bc,)

        # If the boundary segment is axis-aligned, it is straightforward to avoid
        # movement in the normal direction while allowing tangential movement - simply
        # fix the component for the appropriate coordinate
        n = ufl.FacetNormal(self.mesh)
        ds = self.ds(boundary_tag)
        iszero = [np.isclose(firedrake.assemble(abs(ni) * ds), 0.0) for ni in n]
        nzero = sum(iszero)
        assert nzero < self.dim
        if nzero == self.dim - 1:
            idx = iszero.index(False)
            return (firedrake.DirichletBC(self.P1_vec.sub(idx), 0, boundary_tag),)

        # Create an equation boundary condition for enforcing no mesh movement normal to
        # domain boundaries
        u_cts = firedrake.TrialFunction(self.P1_vec)
        v_cts = firedrake.TestFunction(self.P1_vec)
        a_bc = ufl.dot(v_cts, n) * ufl.dot(u_cts, n) * ds
        L_bc = ufl.dot(v_cts, n) * firedrake.Constant(0.0) * ds
        bc1 = firedrake.EquationBC(a_bc == L_bc, self._grad_phi, boundary_tag)

        # Determine the 'corner' vertices which are at the intersection of two boundary
        # segments and create a Dirichlet condition for fixing them under mesh movement
        facet_indices = set(self.mesh.exterior_facets.unique_markers)
        ffacet_indices = [
            (tag, boundary_tag) for tag in facet_indices.difference([boundary_tag])
        ]
        ffacet_bc = firedrake.DirichletBC(self.P1_vec, 0, ffacet_indices)

        # Check the current boundary segment is a straight line or plane
        # TODO: Only do the following check in debugging mode (#94)
        assert self.coord_space.ufl_element().degree() == 1
        ffacets = list(self.mesh.coordinates.dat.data_with_halos[ffacet_bc.nodes])
        hyperplane = equation_of_hyperplane(*ffacets)
        for x in self.mesh.coordinates.dat.data_with_halos[zero_bc.nodes]:
            if not np.isclose(float(hyperplane(*x)), 0):
                raise ValueError(
                    f"Boundary segment '{boundary_tag}' is not"
                    f" {'linear' if self.dim == 2 else 'planar'}."
                )

        # Create an equation boundary condition which allows tangential movement, but
        # only up until the 'corner' vertices where boundary segments meet
        a_bc = ufl.dot(tangential(v_cts, n), tangential(u_cts, n)) * ds
        L_bc = ufl.dot(tangential(v_cts, n), tangential(ufl.grad(self.phi_old), n)) * ds
        bc2 = firedrake.EquationBC(
            a_bc == L_bc, self._grad_phi, boundary_tag, bcs=ffacet_bc
        )
        return bc1, bc2

    @property
    @PETSc.Log.EventDecorator()
    def l2_projector(self):
        """
        Create a linear solver for obtaining the gradient of the potential using an
        :math:`L^2` projection.

        :return: the linear solver
        :rtype: :class:`~.LinearVariationalSolver`
        """
        if hasattr(self, "_l2_projector"):
            return self._l2_projector
        u_cts = firedrake.TrialFunction(self.P1_vec)
        v_cts = firedrake.TestFunction(self.P1_vec)

        # Domain interior
        a = ufl.inner(v_cts, u_cts) * self.dx
        L = ufl.inner(v_cts, ufl.grad(self.phi_old)) * self.dx

        # Enforce no movement normal to boundary
        bcs = [
            dirichlet_bc
            for boundary_tag in self.mesh.exterior_facets.unique_markers
            for dirichlet_bc in self._l2_projector_bcs(boundary_tag)
        ]

        # Check for an attempt to fix periodic boundary segments
        subdomains = set()
        for bc in bcs:
            if not hasattr(bc, "sub_domain"):
                subdomain = bc._F.sub_domain  # EquationBC doesn't have sub_domain attr
            else:
                subdomain = bc.sub_domain
            if not isinstance(subdomain, Iterable):
                subdomain = [subdomain]
            subdomains = subdomains.union(subdomain)
        if set(self.fixed_boundary_segments).difference(subdomains):
            raise ValueError("Cannot fix boundary nodes for periodic segments.")

        # Create solver
        problem = firedrake.LinearVariationalProblem(a, L, self._grad_phi, bcs=bcs)
        self._l2_projector = firedrake.LinearVariationalSolver(
            problem, solver_parameters=solver_parameters.cg_ilu
        )
        return self._l2_projector


class MongeAmpereMover_Relaxation(MongeAmpereMover_Base):
    r"""
    The standard, elliptic form of the Monge-Ampère equation used for mesh movement is:

    .. math::
        m(\mathbf{x})\det(\mathbf{I} + \mathbf{H}(\phi)) = \theta,

    for a convex scalar potential :math:`\phi=\phi(\boldsymbol{\xi})`, which is a
    function of the coordinates of the *computational* mesh. Here :math:`m=m(\mathbf{x})`
    is a user-provided monitor function, which is a function of the coordinates of the
    *physical* mesh. :math:`\mathbf{I}` is the identity matrix, :math:`\theta` is a
    normalisation coefficient  and :math:`\mathbf{H}(\phi)` denotes the Hessian of
    :math:`\phi` with respect to :math:`\boldsymbol{\xi}`.

    In this mesh mover, the Monge-Ampère equation is instead solved in a parabolised form
    using a pseudo-time relaxation,

    .. math::
        -\frac\partial{\partial\tau}\Delta\phi
        = m(\mathbf{x})\det(\mathbf{I} + \mathbf{H}(\phi)) - \theta,

    where :math:`\tau` is the pseudo-time variable. Forward Euler is used for the
    pseudo-time integration (see :cite:`MCB:18` for details).

    This approach typically takes tens or hundreds of iterations to converge, but each
    iteration is relatively cheap.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, mesh, monitor_function, phi_init=None, H_init=None, **kwargs):
        """
        :arg mesh: the physical mesh
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg monitor_function: a Python function which takes a mesh as input
        :type monitor_function: :class:`~.Callable`
        :kwarg phi_init: initial guess for the scalar potential
        :type phi_init: :class:`firedrake.function.Function`
        :kwarg H_init: initial guess for the Hessian
        :type H_init: :class:`firedrake.function.Function`
        :kwarg pseudo_timestep: pseudo-timestep to use for the relaxation
        :type pseudo_timestep: :class:`float`
        :kwarg maxiter: maximum number of iterations for the relaxation
        :type maxiter: :class:`int`
        :kwarg rtol: relative tolerance for the residual
        :type rtol: :class:`float`
        :kwarg dtol: divergence tolerance for the residual
        :type dtol: :class:`float`
        :kwarg fix_boundary_nodes: should all boundary nodes remain fixed?
        :type fix_boundary_nodes: :class:`bool`
        """
        self.pseudo_dt = firedrake.Constant(kwargs.pop("pseudo_timestep", 0.1))
        super().__init__(mesh, monitor_function=monitor_function, **kwargs)

        # Initialise phi and H
        if phi_init or H_init:
            self.apply_initial_guess(phi_init, H_init)

        # Setup residuals
        I = ufl.Identity(self.dim)
        self.theta_form = self.monitor * ufl.det(I + self.H_old) * self.dx
        self.residual = self.monitor * ufl.det(I + self.H_old) - self.theta
        psi = firedrake.TestFunction(self.P1)
        self._residual_l2_form = psi * self.residual * self.dx
        self._norm_l2_form = psi * self.theta * self.dx

    def _create_functions(self):
        super()._create_functions()
        self.phi = firedrake.Function(self.P1)
        self.H = firedrake.Function(self.P1_ten)
        self.phi_old = firedrake.Function(self.P1)
        self.H_old = firedrake.Function(self.P1_ten)

    @property
    @PETSc.Log.EventDecorator()
    def pseudotimestepper(self):
        """
        Setup the pseudo-timestepper for the relaxation method.

        Forward Euler is used for the pseudo-time integration (see :cite:`MCB:18` for
        details). The pseudo-timestep may be set through the `pseudo_timestep` keyword
        argument to the constructor.

        :return: the pseudo-timestepper
        :rtype: :class:`~.LinearVariationalSolver`
        """
        if hasattr(self, "_pseudotimestepper"):
            return self._pseudotimestepper
        phi = firedrake.TrialFunction(self.P1)
        psi = firedrake.TestFunction(self.P1)
        a = ufl.inner(ufl.grad(psi), ufl.grad(phi)) * self.dx
        L = (
            ufl.inner(ufl.grad(psi), ufl.grad(self.phi_old)) * self.dx
            + self.pseudo_dt * psi * self.residual * self.dx
        )
        problem = firedrake.LinearVariationalProblem(a, L, self.phi)
        nullspace = firedrake.VectorSpaceBasis(constant=True)
        self._pseudotimestepper = firedrake.LinearVariationalSolver(
            problem,
            solver_parameters=solver_parameters.cg_gamg,
            nullspace=nullspace,
            transpose_nullspace=nullspace,
        )
        return self._pseudotimestepper

    @property
    @PETSc.Log.EventDecorator()
    def equidistributor(self):
        r"""
        Setup the equidistributor for the relaxation method.

        The equidistributor solves the following equation:

        .. math::
            \int_{\Omega} \tau : \mathbf{H} \, \mathrm{d}x
            = -\int_{\Omega} (\nabla \cdot \tau) \cdot (\nabla \phi) \, \mathrm{d}x
            + \int_{\partial \Omega} ((\nabla \phi \cdot \widehat{\mathbf{n}}) \cdot
            \tau) \cdot \widehat{\mathbf{n}} \, \mathrm{d}s,
            \quad \forall \tau \in \mathbb{P}1^{d \times d}

        for the Hessian :math:`\mathbf{H}`, where :math:`d` is the spatial dimension and
        :math:`\widehat{\mathbf{n}}` is a normal vector to the boundary.

        :return: the equidistributor
        :rtype: :class:`~.LinearVariationalSolver`
        """
        if hasattr(self, "_equidistributor"):
            return self._equidistributor
        n = ufl.FacetNormal(self.mesh)
        tau = firedrake.TestFunction(self.P1_ten)
        a = ufl.inner(tau, firedrake.TrialFunction(self.P1_ten)) * self.dx
        L = (
            -ufl.dot(ufl.div(tau), ufl.grad(self.phi)) * self.dx
            + ufl.dot(ufl.dot(tangential(ufl.grad(self.phi), n), tau), n) * self.ds
        )
        problem = firedrake.LinearVariationalProblem(a, L, self.H)
        self._equidistributor = firedrake.LinearVariationalSolver(
            problem, solver_parameters=solver_parameters.cg_ilu
        )
        return self._equidistributor

    @PETSc.Log.EventDecorator()
    def move(self):
        r"""
        Run the relaxation method to convergence and update the mesh.

        :return: the iteration count
        :rtype: :class:`int`
        """

        # Switch to computational coordinates
        self.to_computational_coordinates()

        # Take iterations of the relaxed system until reaching convergence
        for i in range(self.maxiter):
            self.l2_projector.solve()
            self._update_physical_coordinates()

            # Update monitor function
            self.to_physical_coordinates()
            self.monitor.interpolate(self.monitor_function(self.mesh))
            firedrake.assemble(self.L_P0, tensor=self.volume)
            self.volume.interpolate(self.volume / self.original_volume)
            self.to_computational_coordinates()

            # Evaluate normalisation coefficient
            self.theta.assign(firedrake.assemble(self.theta_form) / self.total_volume)

            # Check convergence criteria
            residual = self.relative_l2_residual
            if i == 0:
                initial_norm = residual
            PETSc.Sys.Print(
                f"{i:4d}"
                f"   Volume ratio {self.volume_ratio:5.2f}"
                f"   Variation (σ/μ) {self.coefficient_of_variation:8.2e}"
                f"   Residual {residual:8.2e}"
            )
            if residual < self.rtol:
                self._convergence_message(i + 1)
                break
            if residual > self.dtol * initial_norm:
                self._divergence_error(i + 1)
            if i == self.maxiter - 1:
                self._convergence_error(i + 1)

            # Apply pseudotimestepper and equidistributor
            self.pseudotimestepper.solve()
            self.equidistributor.solve()
            self.phi_old.assign(self.phi)
            self.H_old.assign(self.H)
        self.to_physical_coordinates()
        return i


class MongeAmpereMover_QuasiNewton(MongeAmpereMover_Base):
    r"""
    The standard, elliptic form of the Monge-Ampère equation used for mesh movement is:

    .. math::
        m(\mathbf{x})\det(\mathbf{I} + \mathbf{H}(\phi)) = \theta,

    for a convex scalar potential :math:`\phi=\phi(\boldsymbol{\xi})`, which is a
    function of the coordinates of the *computational* mesh. Here :math:`m=m(\mathbf{x})`
    is a user-provided monitor function, which is a function of the coordinates of the
    *physical* mesh. :math:`\mathbf{I}` is the identity matrix, :math:`\theta` is a
    normalisation coefficient  and :math:`\mathbf{H}(\phi)` denotes the Hessian of
    :math:`\phi` with respect to :math:`\boldsymbol{\xi}`.

    In this mesh mover, the elliptic Monge-Ampère equation is solved using a quasi-Newton
    method (see :cite:`MCB:18` for details).

    This approach typically takes fewer than ten iterations to converge, but each
    iteration is relatively expensive.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, mesh, monitor_function, phi_init=None, H_init=None, **kwargs):
        """
        :arg mesh: the physical mesh
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg monitor_function: a Python function which takes a mesh as input
        :type monitor_function: :class:`~.Callable`
        :kwarg phi_init: initial guess for the scalar potential
        :type phi_init: :class:`firedrake.function.Function`
        :kwarg H_init: initial guess for the Hessian
        :type H_init: :class:`firedrake.function.Function`
        :kwarg maxiter: maximum number of iterations for the Quasi-Newton solver
        :type maxiter: :class:`int`
        :kwarg rtol: relative tolerance for the residual
        :type rtol: :class:`float`
        :kwarg dtol: divergence tolerance for the residual
        :type dtol: :class:`float`
        """
        super().__init__(mesh, monitor_function=monitor_function, **kwargs)

        # Initialise phi and H
        if phi_init or H_init:
            self.apply_initial_guess(phi_init, H_init)

        # Setup residuals
        I = ufl.Identity(self.dim)
        self.theta_form = self.monitor * ufl.det(I + self.H_old) * self.dx
        self.residual = self.monitor * ufl.det(I + self.H_old) - self.theta
        psi = firedrake.TestFunction(self.P1)
        self._residual_l2_form = psi * self.residual * self.dx
        self._norm_l2_form = psi * self.theta * self.dx

    def _create_functions(self):
        super()._create_functions()
        self.V = self.P1 * self.P1_ten
        self.phi_and_hessian = firedrake.Function(self.V)
        self.phi, self.H = self.phi_and_hessian.subfunctions
        self.phi_and_hessian_old = firedrake.Function(self.V)
        self.phi_old, self.H_old = self.phi_and_hessian_old.subfunctions

    @property
    @PETSc.Log.EventDecorator()
    def equidistributor(self):
        r"""
        Setup the equidistributor for the quasi-newton method.

        The equidistributor solves

        .. math::
            \int_{\Omega} \boldsymbol{\tau} \cdot \mathbf{H} \, \mathrm{d}x
            + \int_{\Omega} (\nabla \cdot \boldsymbol{\tau}) \cdot (\nabla \phi) \,
            \mathrm{d}x
            - \int_{\partial \Omega} (((\nabla \phi) \cdot \widehat{\mathbf{n}}) \cdot
              \boldsymbol{\tau}) \cdot \widehat{\mathbf{n}} \, \mathrm{d}s
            - \int_{\Omega} \psi (m \det(\mathbf{I} + \mathbf{H}) - \theta) \,
              \mathrm{d}x = 0,
              \quad \forall \boldsymbol{\tau} \in \mathbb{P}1^{d \times d},
              \quad \forall \psi \in \mathbb{P}1,

        for the potential :math:`\phi` and its Hessian :math:`\mathbf{H}`, where
        :math:`d` is the spatial dimension and :math:`\widehat{\mathbf{n}}` is a normal
        vector to the boundary.

        :return: the equidistributor
        :rtype: :class:`~.NonlinearVariationalSolver`
        """
        if hasattr(self, "_equidistributor"):
            return self._equidistributor
        n = ufl.FacetNormal(self.mesh)
        I = ufl.Identity(self.dim)
        phi, H = firedrake.split(self.phi_and_hessian)
        psi, tau = firedrake.TestFunctions(self.V)
        F = (
            ufl.inner(tau, H) * self.dx
            + ufl.dot(ufl.div(tau), ufl.grad(phi)) * self.dx
            - ufl.dot(ufl.dot(tangential(ufl.grad(phi), n), tau), n) * self.ds
            - psi * (self.monitor * ufl.det(I + H) - self.theta) * self.dx
        )
        phi, H = firedrake.TrialFunctions(self.V)

        @PETSc.Log.EventDecorator("MongeAmpereMover.update_monitor")
        def update_monitor(cursol):
            """
            Callback for updating the monitor function.
            """
            with self.phi_and_hessian_old.dat.vec as v:
                cursol.copy(v)
            self.l2_projector.solve()
            self._update_physical_coordinates()
            self.to_physical_coordinates()
            self.monitor.interpolate(self.monitor_function(self.mesh))
            self.to_computational_coordinates()
            self.theta.assign(
                firedrake.assemble(self.theta_form) * self.total_volume ** (-1)
            )

        # Setup the variational problem
        # =============================
        # We use a custom preconditioner Jp, chosen to approximate the Jacobian of the
        # system. It includes terms that represent the inner product of tau and H,
        # the product of phi and psi, and the inner product of the gradients of phi and
        # psi. This helps in stabilising the solver and improving convergence.
        Jp = (
            ufl.inner(tau, H) * self.dx
            + phi * psi * self.dx
            + ufl.inner(ufl.grad(phi), ufl.grad(psi)) * self.dx
        )
        problem = firedrake.NonlinearVariationalProblem(F, self.phi_and_hessian, Jp=Jp)

        # Setup the variational solver
        # ============================
        # A nullspace is defined to handle the invariance of the solution under certain
        # transformations. The first component is a constant vector space basis, since
        # constant shifts in phi do not affect the solution.
        nullspace = firedrake.MixedVectorSpaceBasis(
            self.V, [firedrake.VectorSpaceBasis(constant=True), self.V.sub(1)]
        )
        # Note that different solver parameters are used for serial and parallel runs
        sp = (
            solver_parameters.serial_qn
            if firedrake.COMM_WORLD.size == 1
            else solver_parameters.parallel_qn
        )
        sp["snes_atol"] = self.rtol
        sp["snes_max_it"] = self.maxiter
        self._equidistributor = firedrake.NonlinearVariationalSolver(
            problem,
            nullspace=nullspace,
            transpose_nullspace=nullspace,
            pre_function_callback=update_monitor,
            pre_jacobian_callback=update_monitor,
            solver_parameters=sp,
        )

        @no_annotations
        @PETSc.Log.EventDecorator("MongeAmpereMover.callback")
        def callback(snes, i, rnorm):
            """
            Print progress of the optimisation to screen.

            Note that convergence is not actually checked.
            """
            cursol = snes.getSolution()
            update_monitor(cursol)
            self.to_physical_coordinates()
            firedrake.assemble(self.L_P0, tensor=self.volume)
            self.volume.interpolate(self.volume / self.original_volume)
            self.to_computational_coordinates()
            PETSc.Sys.Print(
                f"{i:4d}"
                f"   Volume ratio {self.volume_ratio:5.2f}"
                f"   Variation (σ/μ) {self.coefficient_of_variation:8.2e}"
                f"   Residual {self.relative_l2_residual:8.2e}"
            )

        self.snes = self._equidistributor.snes
        self.snes.setMonitor(callback)
        return self._equidistributor

    @PETSc.Log.EventDecorator()
    def move(self):
        r"""
        Run the quasi-Newton method to convergence and update the mesh.

        :return: the iteration count
        :rtype: :class:`int`
        """
        # Solve equidistribution problem, handling convergence errors according to
        # desired behaviour
        try:
            self.equidistributor.solve()
            self._convergence_message(self.snes.getIterationNumber())
        except fexc.ConvergenceError as conv_err:
            self._convergence_error(self.snes.getIterationNumber(), exception=conv_err)

        # Update mesh coordinates accordingly
        self.to_physical_coordinates()
        return self.snes.getIterationNumber()
