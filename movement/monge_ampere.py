import firedrake
from movement.mesh_mover import MeshMover


class MongeAmpereMover(MeshMover):
    # TODO: doc
    def __init__(self, mesh, monitor_function, **kwargs):
        if monitor_function is None:
            raise ValueError("Please supply a monitor function")
        super().__init__(mesh, monitor_function=monitor_function)

        # Collect parameters
        self.pseudo_dt = firedrake.Constant(kwargs.get('pseudo_timestep', 0.1))
        self.bc = kwargs.get('boundary_conditions', None)

        # Create function spaces
        self.P0 = firedrake.FunctionSpace(mesh, "DG", 0)
        self.P1 = firedrake.FunctionSpace(mesh, "CG", 1)
        self.P1_vec = firedrake.VectorFunctionSpace(mesh, "CG", 1)
        self.P1_ten = firedrake.TensorFunctionSpace(mesh, "CG", 1)

        # Create objects used during the mesh movement
        self.dx = firedrake.dx(domain=mesh)
        self.theta = firedrake.Constant(0.0)
        self._monitor = firedrake.Function(self.P1, name="Monitor function")
        self.volume = firedrake.Function(self.P0, name="Mesh volume")
        self.volume.interpolate(CellVolume(mesh))
        self.original_volume = firedrake.Function(self.volume)
        self.total_volume = firedrake.assemble(firedrake.Constant(1.0)*self.dx)
        self.L_P0 = firedrake.TestFunction(self.P0)*self.monitor*self.dx
        self.grad_phi_cg = firedrake.Function(self.P1_vec)
        self.grad_phi_dg = firedrake.Function(self.mesh.coordinates)

    @property
    def monitor(self):
        """
        Update the monitor function based on the current mesh.
        """
        self._monitor.interpolate(self.monitor_function(self.mesh))
        return self._monitor


def monge_ampere(mesh, monitor_function):
    mover = MeshMover(mesh, monitor_function)
    mover.adapt()
