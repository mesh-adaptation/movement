import firedrake
from firedrake import PETSc
import ufl
import numpy as np
from movement.mover import Mover


__all__ = ["MongeAmpereMover", "monge_ampere"]


class MongeAmpereMover_Base(Mover):
    # TODO: docs
    residual_l2_form = 0
    norm_l2_form = 0

    def __init__(self, mesh, monitor_function, **kwargs):
        """
        :arg mesh: the physical mesh
        :arg monitor_function: a Python function which takes a mesh as input
        :kwarg maxiter: maximum number of iterations for the relaxation
        :kwarg rtol: relative tolerance for the residual
        :kwarg dtol: divergence tolerance for the residual
        """
        if monitor_function is None:
            raise ValueError("Please supply a monitor function")

        # Collect parameters before calling super
        self.pseudo_dt = firedrake.Constant(kwargs.pop('pseudo_timestep', 0.1))
        self.maxiter = kwargs.pop('maxiter', 1000)
        self.rtol = kwargs.pop('rtol', 1.0e-08)
        self.dtol = kwargs.pop('dtol', 2.0)
        super().__init__(mesh, monitor_function=monitor_function)

        # Create function spaces
        self.P0 = firedrake.FunctionSpace(mesh, "DG", 0)
        self.P1 = firedrake.FunctionSpace(mesh, "CG", 1)
        self.P1_vec = firedrake.VectorFunctionSpace(mesh, "CG", 1)
        self.P1_ten = firedrake.TensorFunctionSpace(mesh, "CG", 1)

        # Create objects used during the mesh movement
        self.theta = firedrake.Constant(0.0)
        self.monitor = firedrake.Function(self.P1, name="Monitor function")
        self.monitor.interpolate(self.monitor_function(self.mesh))
        self.volume = firedrake.Function(self.P0, name="Mesh volume")
        self.volume.interpolate(ufl.CellVolume(mesh))
        self.original_volume = firedrake.Function(self.volume)
        self.total_volume = firedrake.assemble(firedrake.Constant(1.0)*self.dx)
        self.L_P0 = firedrake.TestFunction(self.P0)*self.monitor*self.dx
        self._grad_phi = firedrake.Function(self.P1_vec)
        self.grad_phi = firedrake.Function(self.mesh.coordinates)

    @property
    def diagnostics(self):
        """
        Compute diagnostics:
          1) the ratio of the smallest and largest element volumes;
          2) equidistribution of elemental volumes;
          3) relative L2 norm residual.
        """
        v = self.volume.vector().gather()
        minmax = v.min()/v.max()
        mean = v.sum()/v.max()
        w = v.copy() - mean
        w *= w
        std = np.sqrt(w.sum()/w.size)
        equi = std/mean
        residual_l2 = firedrake.assemble(self.residual_l2_form).dat.norm
        norm_l2 = firedrake.assemble(self.norm_l2_form).dat.norm
        residual_l2_rel = residual_l2/norm_l2
        return minmax, residual_l2_rel, equi

    @property
    def x(self):
        """
        Update the coordinate :class:`Function` using
        the recovered gradient.
        """
        try:
            self.grad_phi.assign(self._grad_phi)
        except Exception:
            self.grad_phi.interpolate(self._grad_phi)
        self._x.assign(self.xi + self.grad_phi)  # x = ξ + grad(φ)
        return self._x

    @property
    def l2_projector(self):
        """
        Create a linear solver for obtaining the gradient
        of the potential using an L2 projection.

        Boundary conditions are imposed as a post-processing step.
        """
        if hasattr(self, '_l2_projector'):
            return self._l2_projector
        u_cts = firedrake.TrialFunction(self.P1_vec)
        v_cts = firedrake.TestFunction(self.P1_vec)

        # Domain interior
        a = ufl.inner(v_cts, u_cts)*self.dx
        L = ufl.inner(v_cts, ufl.grad(self.phi_old))*self.dx

        # Enforce no movement normal to boundary
        n = ufl.FacetNormal(self.mesh)
        bcs = []
        for i in self.mesh.exterior_facets.unique_markers:
            _n = [firedrake.assemble(n[j]*self.ds(i)) for j in range(self.dim)]
            if np.allclose(_n, 0.0):
                raise ValueError(f"Invalid normal vector {_n}")
            else:
                if self.dim != 2:
                    raise NotImplementedError  # TODO
                if np.isclose(_n[0], 0.0):
                    bcs.append(firedrake.DirichletBC(self.P1_vec.sub(1), 0, i))
                elif np.isclose(_n[1], 0.0):
                    bcs.append(firedrake.DirichletBC(self.P1_vec.sub(0), 0, i))
                else:
                    # Enforce no mesh movement normal to boundaries
                    a_bc = ufl.dot(v_cts, n)*ufl.dot(u_cts, n)*self.ds
                    L_bc = ufl.dot(v_cts, n)*firedrake.Constant(0.0)*self.ds
                    bcs.append(firedrake.EquationBC(a_bc == L_bc, self.grad_phi, 'on_boundary'))

                    # Allow tangential movement, but only up until the end of boundary segments
                    s = ufl.perp(n)
                    a_bc = ufl.dot(v_cts, s)*ufl.dot(u_cts, s)*self.ds
                    L_bc = ufl.dot(v_cts, s)*ufl.dot(ufl.grad(self.phi_old), s)*self.ds
                    edges = set(self.mesh.exterior_facets.unique_markers)
                    if len(edges) > 1:  # NOTE: Assumes that all straight line segments are uniquely tagged
                        corners = [(i, j) for i in edges for j in edges.difference([i])]
                        bbc = firedrake.DirichletBC(self.P1_vec, 0, corners)
                    else:
                        bbc = None
                    bcs.append(firedrake.EquationBC(a_bc == L_bc, self.grad_phi, 'on_boundary', bcs=bbc))

        # Create solver
        problem = firedrake.LinearVariationalProblem(a, L, self._grad_phi, bcs=bcs)
        sp = {"ksp_type": "cg"}
        self._l2_projector = firedrake.LinearVariationalSolver(problem, solver_parameters=sp)
        return self._l2_projector


class MongeAmpereMover_Relaxation(MongeAmpereMover_Base):
    r"""
    Movement of a `mesh` is determined by a `monitor_function`
    :math:`m` and the Monge-Ampère type equation

..  math::
        m(x)\det(I + H(\phi)) = \theta,

    for a scalar potential :math:`\phi`, where :math:`I` is the
    identity matrix, :math:`\theta` is a normalisation coefficient
    and :math:`H(\phi)` denotes the Hessian of :math:`\phi` with
    respect to the coordinates :math:`\xi` of the computational mesh.

    The physical mesh coordinates :math:`x` are updated according to

..  math::
        x = \xi + \nabla\phi.
    """
    # TODO: docs specific to relaxation
    def __init__(self, mesh, monitor_function, **kwargs):
        """
        :arg mesh: the physical mesh
        :arg monitor_function: a Python function which takes a mesh as input
        :kwarg pseudo_timestep: pseudo-timestep to use for the relaxation
        :kwarg maxiter: maximum number of iterations for the relaxation
        :kwarg rtol: relative tolerance for the residual
        :kwarg dtol: divergence tolerance for the residual
        """
        self.pseudo_dt = firedrake.Constant(kwargs.pop('pseudo_timestep', 0.1))
        super().__init__(mesh, monitor_function=monitor_function, **kwargs)

        # Create functions to hold solution data
        self.phi = firedrake.Function(self.P1)        # NOTE: initialised to zero
        self.sigma = firedrake.Function(self.P1_ten)  # NOTE: initialised to zero
        self.phi_old = firedrake.Function(self.P1)
        self.sigma_old = firedrake.Function(self.P1_ten)

        # Setup residuals
        I = ufl.Identity(self.dim)
        self.theta_form = self.monitor*ufl.det(I + self.sigma_old)*self.dx
        self.residual = self.monitor*ufl.det(I + self.sigma_old) - self.theta
        psi = firedrake.TestFunction(self.P1)
        self.residual_l2_form = psi*self.residual*self.dx
        self.norm_l2_form = psi*self.theta*self.dx

    @property
    def pseudotimestepper(self):
        """
        Setup the pseudo-timestepper for the relaxation method.
        """
        if hasattr(self, '_pseudotimestepper'):
            return self._pseudotimestepper
        phi = firedrake.TrialFunction(self.P1)
        psi = firedrake.TestFunction(self.P1)
        a = ufl.inner(ufl.grad(psi), ufl.grad(phi))*self.dx
        L = ufl.inner(ufl.grad(psi), ufl.grad(self.phi_old))*self.dx \
            + self.pseudo_dt*psi*self.residual*self.dx
        problem = firedrake.LinearVariationalProblem(a, L, self.phi)
        sp = {
            "ksp_type": "cg",
            "pc_type": "gamg",
        }
        nullspace = firedrake.VectorSpaceBasis(constant=True)
        self._pseudotimestepper = firedrake.LinearVariationalSolver(
            problem, solver_parameters=sp,
            nullspace=nullspace, transpose_nullspace=nullspace,
        )
        return self._pseudotimestepper

    @property
    def equidistributor(self):
        """
        Setup the equidistributor for the relaxation method.
        """
        if hasattr(self, '_equidistributor'):
            return self._equidistributor
        if self.dim != 2:
            raise NotImplementedError  # TODO
        n = ufl.FacetNormal(self.mesh)
        sigma = firedrake.TrialFunction(self.P1_ten)
        tau = firedrake.TestFunction(self.P1_ten)
        a = ufl.inner(tau, sigma)*self.dx
        L = -ufl.dot(ufl.div(tau), ufl.grad(self.phi))*self.dx \
            + (tau[0, 1]*n[1]*self.phi.dx(0) + tau[1, 0]*n[0]*self.phi.dx(1))*self.ds
        problem = firedrake.LinearVariationalProblem(a, L, self.sigma)
        sp = {"ksp_type": "cg"}
        self._equidistributor = firedrake.LinearVariationalSolver(problem, solver_parameters=sp)
        return self._equidistributor

    def adapt(self):
        """
        Run the relaxation method to convergence and update the mesh.
        """
        for i in range(self.maxiter):

            # L2 project
            self.l2_projector.solve()

            # Update mesh coordinates
            self.mesh.coordinates.assign(self.x)

            # Update monitor function
            self.monitor.interpolate(self.monitor_function(self.mesh))
            firedrake.assemble(self.L_P0, tensor=self.volume)
            self.volume.assign(self.volume*self.original_volume**(-1))
            self.mesh.coordinates.assign(self.xi)

            # Evaluate normalisation coefficient
            self.theta.assign(firedrake.assemble(self.theta_form)*self.total_volume**(-1))

            # Check convergence criteria
            minmax, residual, equi = self.diagnostics
            if i == 0:
                initial_norm = residual
            PETSc.Sys.Print(f"{i:4d}"
                            f"   Min/Max {minmax:10.4e}"
                            f"   Residual {residual:10.4e}"
                            f"   Equidistribution {equi:10.4e}")
            if residual < self.rtol:
                PETSc.Sys.Print(f"Converged in {i+1} iterations.")
                break
            if residual > self.dtol*initial_norm:
                raise firedrake.ConvergenceError(f"Diverged after {i+1} iterations.")
            if i == self.maxiter-1:
                raise firedrake.ConvergenceError(f"Failed to converge in {i+1} iterations.")

            # Apply pseudotimestepper and equidistributor
            self.pseudotimestepper.solve()
            self.equidistributor.solve()
            self.phi_old.assign(self.phi)
            self.sigma_old.assign(self.sigma)
        self.mesh.coordinates.assign(self.x)


class MongeAmpereMover_QuasiNewton(Mover):
    # TODO: docs specific to qn
    def __init__(self, mesh, monitor_function, **kwargs):
        """
        :arg mesh: the physical mesh
        :arg monitor_function: a Python function which takes a mesh as input
        :kwarg maxiter: maximum number of iterations for the relaxation
        :kwarg rtol: relative tolerance for the residual
        :kwarg dtol: divergence tolerance for the residual
        """
        super().__init__(mesh, monitor_function=monitor_function, **kwargs)

        # Create functions to hold solution data
        self.V = self.P1*self.P1_ten
        self.phisigma = firedrake.Function(self.V)  # NOTE: initialised to zero
        self.phi, self.sigma = self.phisigma.split()
        self.phisigma_old = firedrake.Function(self.V)
        self.phi_old, self.sigma_old = self.phisigma_old.split()

        # Setup residuals
        I = ufl.Identity(self.dim)
        self.theta_form = self.monitor*ufl.det(I + self.sigma_old)*self.dx
        self.residual = self.monitor*ufl.det(I + self.sigma_old) - self.theta
        psi = firedrake.TestFunction(self.P1)
        self.residual_l2_form = psi*self.residual*self.dx
        self.norm_l2_form = psi*self.theta*self.dx


def monge_ampere(mesh, monitor_function, method='relaxation', **kwargs):
    r"""
    Movement of a `mesh` is determined by a `monitor_function`
    :math:`m` and the Monge-Ampère type equation

..  math::
        m(x)\det(I + H(\phi)) = \theta,

    for a scalar potential :math:`\phi`, where :math:`I` is the
    identity matrix, :math:`\theta` is a normalisation coefficient
    and :math:`H(\phi)` denotes the Hessian of :math:`\phi` with
    respect to the coordinates :math:`\xi` of the computational mesh.

    The physical mesh coordinates :math:`x` are updated according to

..  math::
        x = \xi + \nabla\phi.

    :arg mesh: the physical mesh
    :arg monitor_function: a Python function which takes a mesh as input
    :kwarg method: choose from 'relaxation' and 'quasi_newton'
    """
    if method == 'relaxation':
        mover = MongeAmpereMover_Relaxation(mesh, monitor_function, **kwargs)
    elif method == 'quasi_newton':
        mover = MongeAmpereMover_QuasiNewton(mesh, monitor_function, **kwargs)
    else:
        raise ValueError(f"Monge-Ampere solver {method} not recognised.")
    mover.adapt()
