serial_qn = {
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "pc_fieldsplit_off_diag_use_amat": True,
    "fieldsplit_0_pc_type": "gamg",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_mg_levels_ksp_max_it": 5,
    "fieldsplit_0_mg_levels_pc_type": "ilu",
    "fieldsplit_1_pc_type": "ilu",
    "fieldsplit_1_ksp_type": "preonly",
    "ksp_max_it": 200,
    "snes_max_it": 125,
    "ksp_gmres_restart": 200,
    "snes_rtol": 1.0e-08,
    "snes_linesearch_type": "l2",
    "snes_linesearch_max_it": 5,
    "snes_linesearch_maxstep": 1.05,
    "snes_linesearch_damping": 0.8,
    "snes_lag_preconditioner": -2,
}

parallel_qn = {
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "pc_fieldsplit_off_diag_use_amat": True,
    "fieldsplit_0_pc_type": "gamg",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_mg_levels_ksp_max_it": 5,
    "fieldsplit_0_mg_levels_pc_type": "bjacobi",
    "fieldsplit_0_mg_levels_sub_ksp_type": "preonly",
    "fieldsplit_0_mg_levels_sub_pc_type": "ilu",
    "fieldsplit_1_pc_type": "bjacobi",
    "fieldsplit_1_sub_ksp_type": "preonly",
    "fieldsplit_1_sub_pc_type": "ilu",
    "fieldsplit_1_ksp_type": "preonly",
    "ksp_max_it": 200,
    "snes_max_it": 125,
    "ksp_gmres_restart": 200,
    "snes_rtol": "1.0e-08",
    "snes_linesearch_type": "l2",
    "snes_linesearch_max_it": 5,
    "snes_linesearch_maxstep": 1.05,
    "snes_linesearch_damping": 0.8,
    "snes_lag_preconditioner": -2,
}

mass_inv = {
    "mat_type": "matfree",
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.MassInvPC",
}

jacobi = {
    "mat_type": "matfree",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "jacobi",
}

cg = {
    "ksp_type": "cg",
    "pc_type": "bjacobi",
    "pc_sub_type": "ilu",
}
