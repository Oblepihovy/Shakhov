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

matplotlib.use('TkAgg')

#from src.datio import write_to_csv

from src.config.libloader import xp, cuda_is_available


t_max = 0.2
TD_KN = 1e-5

n_x = 80
n_xi = 20




model_config = {'X_LEFT': X_LEFT, 'X_RIGHT': X_RIGHT, 'n_x': n_x,
                'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
                'F_BEG_N': F_BEG_N, 'F_BEG_U': F_BEG_U, 'F_BEG_T': F_BEG_T,
                'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W, 'dtype': dtype}


bc = ZeroGradBoundaryCondition(2)
#bc = EvapCondBoundaryCondition(2, lambda t: 5., lambda t: 5.)
#mesh1 = UnadaptableMesh(graded_linspace(xp, n_points=n_x, a=0.01, length=X_RIGHT), bc.n_ghost)

#mesh1 = RezoningMesh(xp.linspace(X_LEFT, X_RIGHT, n_x, endpoint=True), bc.n_ghost, alpha=0.9)
mesh1 = UnadaptableMesh(xp.linspace(X_LEFT, X_RIGHT, n_x, endpoint=True), bc.n_ghost)

CFL = 2

adv_solver = SolverDTSS2(
    explicit_solver=SolverKolgan(),
    n_iter=6,
    omega=0.7,
    cfl_pseudo=0.9
)
#adv_solver = SolverRK()
properties = ModelProperties(model_config, mesh1, bc)
state = ModelState(properties, model_config)
solver = ShakhovSolver(state, properties, adv_solver)
t1 = time.time()
solver.calculate(CFL, t_max)
t2 = time.time()
print("S1 calculation time = ", t2-t1)



x = properties.mesh.get_centers()[bc.n_ghost:len(properties.mesh.x) - bc.n_ghost + 1]


if cuda_is_available:
    x = xp.asnumpy(x)
n, u, T, q = PropertyCalculator.get_solution_macros(state.F, properties)

fig, axs = plt.subplots(1, 3)
fig.suptitle(f'{adv_solver.get_name()}, n_x:{n_x}, x:({X_LEFT},{X_RIGHT},{n_x}), xi:({XI_LEFT},{XI_RIGHT},{n_xi}), t:{t_max.__round__(3)}, CFL:{CFL}, Kn:{TD_KN}')

print(x.shape, n.shape)

axs[0].set_title('n (density)')
axs[0].scatter(x, n, linewidth=0.01)
axs[0].plot(x, n, color='blue', label=f'{adv_solver.get_name()}, n_x={n_x}')
axs[0].grid()

axs[1].set_title('u (velocity)')
axs[1].scatter(x, u, linewidth=0.01)
axs[1].plot(x, u, color='blue', label=f'{adv_solver.get_name()}, n_x={n_x}')
axs[1].grid()

axs[2].set_title('T (temperature)')
axs[2].scatter(x, T, linewidth=0.01)
axs[2].plot(x, T, color='blue', label=f'{adv_solver.get_name()}, n_x={n_x}')
axs[2].grid()


#path = 'Tolstyh2'
#plt.savefig(f'infographics/{path}/n_x:{n_x}_xi:({XI_LEFT},{XI_RIGHT},{n_xi})_t:{t_max}_CFL:{CFL}_Kn:{TD_KN}.png', dpi=300)
#write_to_csv(x, n, u, T, q, f'calculated_data/{path}/n_x:{n_x}_xi:({XI_LEFT},{XI_RIGHT},{n_xi})_t:{t_max}_CFL:{CFL}_Kn:{TD_KN}.dat')

CFL=0.8
adv_solver = SolverRK()
"""CFL = 3
adv_solver = SolverDTSS2(
    explicit_solver=SolverKolgan(),
    n_iter=5,
    omega=0.7,
    cfl_pseudo=0.9
)"""
mesh2 = UnadaptableMesh(xp.linspace(X_LEFT, X_RIGHT, n_x, endpoint=True), bc.n_ghost)
properties = ModelProperties(model_config, mesh2, bc)
state = ModelState(properties, model_config)
solver = ShakhovSolver(state, properties, adv_solver)

t1 = time.time()
solver.calculate(CFL, t_max)
t2 = time.time()
print("S2 calculation time = ", t2-t1)

#x = properties.mesh.x[bc.n_ghost:len(properties.mesh.x)-bc.n_ghost+1]+properties.mesh.h/2
x2 = properties.mesh.get_centers()[bc.n_ghost:len(properties.mesh.x) - bc.n_ghost + 1]
if cuda_is_available:
    x = xp.asnumpy(x)
n2, u2, T2, q2 = PropertyCalculator.get_solution_macros(state.F, properties)




axs[0].scatter(x2, n2, linewidth=0.01)
axs[0].plot(x2, n2, color='red', label=f'{adv_solver.get_name()}, n_x={n_x}')
#axs[0].grid()

axs[1].scatter(x2, u2, linewidth=0.01)
axs[1].plot(x2, u2, color='red', label=f'{adv_solver.get_name()}, n_x={n_x}')
#axs[1].grid()

axs[2].scatter(x2, T2, linewidth=0.01)
axs[2].plot(x2, T2, color='red', label=f'{adv_solver.get_name()}, n_x={n_x}')
#axs[2].grid()


adv_solver = WENO5RK3()
mesh3 = UnadaptableMesh(xp.linspace(X_LEFT, X_RIGHT, n_x, endpoint=True), bc.n_ghost)
properties = ModelProperties(model_config, mesh3, bc)
state = ModelState(properties, model_config)
solver = ShakhovSolver(state, properties, adv_solver)

t1 = time.time()
solver.calculate(CFL, t_max)
t2 = time.time()
print("S3 calculation time = ", t2-t1)



#x = properties.mesh.x[bc.n_ghost:len(properties.mesh.x)-bc.n_ghost+1]+properties.mesh.h/2
x3 = properties.mesh.get_centers()[bc.n_ghost:len(properties.mesh.x) - bc.n_ghost + 1]
if cuda_is_available:
    x = xp.asnumpy(x)
n3, u3, T3, q3 = PropertyCalculator.get_solution_macros(state.F, properties)




axs[0].scatter(x3, n3, linewidth=0.01)
axs[0].plot(x3, n3, color='green', label=f'{adv_solver.get_name()}, n_x={n_x}')
#axs[0].grid()

axs[1].scatter(x3, u3, linewidth=0.01)
axs[1].plot(x3, u3, color='green', label=f'{adv_solver.get_name()}, n_x={n_x}')
#axs[1].grid()

axs[2].scatter(x3, T3, linewidth=0.01)
axs[2].plot(x3, T3, color='green', label=f'{adv_solver.get_name()}, n_x={n_x}')
#axs[2].grid()


CFL=0.8
adv_solver = SolverGodunov()

mesh4 = UnadaptableMesh(xp.linspace(X_LEFT, X_RIGHT, n_x, endpoint=True), bc.n_ghost)
properties = ModelProperties(model_config, mesh4, bc)
state = ModelState(properties, model_config)
solver = ShakhovSolver(state, properties, adv_solver)

t1 = time.time()
solver.calculate(CFL, t_max)
t2 = time.time()
print("S2 calculation time = ", t2-t1)

#x = properties.mesh.x[bc.n_ghost:len(properties.mesh.x)-bc.n_ghost+1]+properties.mesh.h/2
x4 = properties.mesh.get_centers()[bc.n_ghost:len(properties.mesh.x) - bc.n_ghost + 1]
if cuda_is_available:
    x = xp.asnumpy(x)
n4, u4, T4, q4 = PropertyCalculator.get_solution_macros(state.F, properties)




axs[0].scatter(x4, n4, linewidth=0.01)
axs[0].plot(x4, n4, color='purple', label=f'{adv_solver.get_name()}, n_x={n_x}')
#axs[0].grid()c

axs[1].scatter(x4, u4, linewidth=0.01)
axs[1].plot(x4, u4, color='purple', label=f'{adv_solver.get_name()}, n_x={n_x}')
#axs[1].grid()

axs[2].scatter(x4, T4, linewidth=0.01)
axs[2].plot(x4, T4, color='purple', label=f'{adv_solver.get_name()}, n_x={n_x}')


axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.show()
