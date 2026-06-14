import time

from src.advection_solvers.DTSS1 import SolverDTSS1
from src.advection_solvers.DTSS2 import SolverDTSS2
from src.advection_solvers.WENO5RK3 import WENO5RK3
from src.advection_solvers.rk2 import SolverRK
from src.thermodynamics.boundary_condition import EvapCondBoundaryCondition, ZeroGradBoundaryCondition
from src.config.configuration import *
from src.mesh import UnadaptableMesh, RezoningMesh
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

#from src.datio import write_to_csv

from src.config.libloader import xp, cuda_is_available

CFL = 0.8
t_max = 0.2
TD_KN = 1e-5

n_x = 20
n_xi = 20

#F_BEG_N = lambda x: 1.
#F_BEG_U = lambda x: 0.
#F_BEG_T = lambda x: 1.


model_config = {'X_LEFT': X_LEFT, 'X_RIGHT': X_RIGHT, 'n_x': n_x,
                'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
                'F_BEG_N': F_BEG_N, 'F_BEG_U': F_BEG_U, 'F_BEG_T': F_BEG_T,
                'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W, 'dtype': dtype}


bc = ZeroGradBoundaryCondition(2)
#bc = EvapCondBoundaryCondition(2, lambda t: 5., lambda t: 5.)
#mesh1 = UnadaptableMesh(graded_linspace(xp, n_points=n_x, a=0.01, length=X_RIGHT), bc.n_ghost)

#mesh1 = RezoningMesh(xp.linspace(X_LEFT, X_RIGHT, n_x, endpoint=True), bc.n_ghost, alpha=0.9)
mesh1 = UnadaptableMesh(xp.linspace(X_LEFT, X_RIGHT, n_x, endpoint=True), bc.n_ghost)

"""
adv_solver = SolverDTSS2(
    explicit_solver=SolverKolgan(),
    n_iter=20,
    omega=0.1,
    cfl_pseudo=0.9
)"""
adv_solver = SolverRK()
properties = ModelProperties(model_config, mesh1, bc)
state = ModelState(properties, model_config)
solver = ShakhovSolver(state, properties, adv_solver)
t1 = time.time()
solver.calculate(CFL, t_max)
t2 = time.time()
print("S1 calculation time = ", t2-t1)



#x = properties.mesh.get_centers()[bc.n_ghost:len(properties.mesh.x) - bc.n_ghost + 1]
x = properties.mesh.get_centers()[bc.n_ghost:-bc.n_ghost]



if cuda_is_available:
    x = xp.asnumpy(x)
n, u, T, q = PropertyCalculator.get_solution_macros(state.F, properties)

n_exact, u_exact, T_exact = euler_exact(x, 0.5, 1, 0.125, 0, 0, 1./2., 0.8/2., t_max, gamma=5./3.)
print(f'L2: {L2(x, n_exact, n)}, L1: {L1(x, n_exact, n)}, max: {L_sup(x, n_exact, n)}, mean: {xp.mean(n_exact-n)}')

fig, axs = plt.subplots(1, 3)
fig.suptitle(f'{adv_solver.get_name()}, n_x:{n_x}, x:({X_LEFT},{X_RIGHT},{n_x}), xi:({XI_LEFT},{XI_RIGHT},{n_xi}), t:{t_max.__round__(3)}, CFL:{CFL}, Kn:{TD_KN}')

print(x.shape, n.shape)

axs[0].set_title('n (density)')
axs[0].scatter(x, n, linewidth=0.01)
axs[0].plot(x, n, color='blue', label=f'{adv_solver.get_name()}, n_x={n_x}')
axs[0].plot(x, n_exact, color='black', label=f'exact, n_x={n_x}')
axs[0].grid()

axs[1].set_title('u (velocity)')
axs[1].scatter(x, u, linewidth=0.01)
axs[1].plot(x, u, color='blue', label=f'{adv_solver.get_name()}, n_x={n_x}')
axs[1].plot(x, u_exact, color='black', label=f'exact, n_x={n_x}')
axs[1].grid()

axs[2].set_title('T (temperature)')
axs[2].scatter(x, T, linewidth=0.01)
axs[2].plot(x, T, color='blue', label=f'{adv_solver.get_name()}, n_x={n_x}')
axs[2].plot(x, T_exact*2, color='black', label=f'exact, n_x={n_x}')
axs[2].grid()

plt.show()