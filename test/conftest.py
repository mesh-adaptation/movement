"""
Global pytest configuration.
"""

import pytest
from pyop2.mpi import COMM_WORLD
import numpy as np

np.random.seed(0)


def pytest_configure(config):
    """
    Register an additional marker.
    """
    config.addinivalue_line(
        "markers", 
        "parallel_dynamic: mark test to run with nprocs equal to the current MPI size"
    )

def pytest_collection_modifyitems(config, items):
    """
    Replace ``parallel_dynamic`` markers with mpi-pytest's ``parallel(nprocs=N)``
    marker, where N is the current MPI size determined from the ``mpiexec -n N`` command
    line argument.
    """
    rank_size = COMM_WORLD.Get_size()
    
    for item in items:
        # Check if a test is marked with the parallel_dynamic marker
        markers = [marker for marker in item.own_markers 
                  if marker.name == 'parallel_dynamic']
        
        if markers:
            # Remove parallel_dynamic markers
            for marker in markers:
                item.own_markers.remove(marker)
            
            # Add mpi-pytest's parallel marker with current process count
            item.add_marker(pytest.mark.parallel(nprocs=rank_size))
