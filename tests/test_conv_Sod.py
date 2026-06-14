import time

import numpy as np

from src.advection_solvers.rk2 import SolverRK
from src.thermodynamics.boundary_condition import EvapCondBoundaryCondition, ZeroGradBoundaryCondition
from src.config.configuration import *
from src.mesh import UnadaptableMesh
from src.advection_solvers.godunov import SolverGodunov
from src.advection_solvers.kolgan import SolverKolgan

import matplotlib
import matplotlib.pyplot as plt

from src.thermodynamics.model_properties import ModelProperties
from src.thermodynamics.model_state import ModelState
from src.thermodynamics.property_calculator import PropertyCalculator
from src.thermodynamics.shakhov_solver import ShakhovSolver
from src.utils.sod_exact import euler_exact, L2, L_sup, L1

matplotlib.use('TkAgg')


from src.config.libloader import xp, cuda_is_available

def F_BEG_N(x):
    return xp.where(x <= 0.5, 1., 0.125)

def F_BEG_U(x):
    return xp.zeros_like(x)

def F_BEG_T(x):
    return xp.where(x <= 0.5, 1., 0.8)


def get_Sod_convergence(advection_solvers, n_x_vals, n_xi_vals):
    """

    :param advection_solvers:
    :param n_x_vals:
    :param n_xi_vals:
    :return: Errors: len(n_x_vals) x 3, times: len(n_x_vals)
    """

    CFL = 0.8
    t_max = 0.2
    TD_KN = 1e-5
    dtype = xp.float32

    calculation_times = np.zeros_like(n_x_vals, dtype=dtype)
    calculation_errors = np.zeros((len(n_x_vals), 3))

    if len(n_x_vals) != len(n_xi_vals):
        print('n_x_vals and n_xi_vals must have the same length')
        return -1

    for i in range(len(n_x_vals)):

        n_x = n_x_vals[i]
        n_xi = n_xi_vals[i]

        model_config = {'X_LEFT': X_LEFT, 'X_RIGHT': X_RIGHT, 'n_x': n_x,
                        'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
                        'F_BEG_N': F_BEG_N, 'F_BEG_U': F_BEG_U, 'F_BEG_T': F_BEG_T,
                        'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W, 'dtype': dtype}

        bc = ZeroGradBoundaryCondition(2)
        mesh1 = UnadaptableMesh(xp.linspace(X_LEFT, X_RIGHT, n_x, endpoint=True), bc.n_ghost)

        adv_solver = advection_solvers[i]()
        properties = ModelProperties(model_config, mesh1, bc)
        state = ModelState(properties, model_config)
        solver = ShakhovSolver(state, properties, adv_solver)
        t1 = time.time()
        solver.calculate(CFL, t_max)
        t2 = time.time()
        print("S1 calculation time = ", t2 - t1)
        calculation_times[i] = t2 - t1

        x = properties.mesh.get_centers()[bc.n_ghost:-bc.n_ghost]
        if cuda_is_available:
            x = xp.asnumpy(x)
        n, u, T, q = PropertyCalculator.get_solution_macros(state.F, properties)
        n_exact, u_exact, T_exact = euler_exact(x, 0.5, 1, 0.125, 0, 0, 1. / 2., 0.8 / 2., t_max, gamma=5. / 3.)


        print(
            f'solver: {advection_solvers[i].get_name()}' 
            f'L2: {L2(x, n_exact, n)},'
            f' L1: {L1(x, n_exact, n)},'
            f' max: {L_sup(x, n_exact, n)},'
            f' mean: {xp.mean(n_exact - n)}')
        calculation_errors[i][0] = L_sup(x, n_exact, n)
        calculation_errors[i][1] = L1(x, n_exact, n)
        calculation_errors[i][2] = L2(x, n_exact, n)

    return calculation_errors, calculation_times

L, t_calc = get_Sod_convergence([SolverRK]*5, [10, 20, 40, 80, 160], [20, 20, 20, 20, 20])
for i in range(3):
    print(f'L{i}: {L[:, i]}')
print(f't_calc: {t_calc}')
