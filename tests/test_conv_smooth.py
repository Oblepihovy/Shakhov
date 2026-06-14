"""
test_conv_smooth.py
-------------------
Тест сходимости на гладком решении — бегущая акустическая волна.

Начальное условие: изентропически согласованное синусоидальное возмущение.
Граничные условия: периодические.
Точное решение: уравнения Эйлера (работает при Kn << 1).

Ожидаемые порядки по L2:
  Godunov            ~1
  Kolgan / RK2       ~2
  WENO5 + SSP-RK3    ~3  (τ ~ h → временна́я ошибка O(h³) доминирует над O(h⁵))

Требования к окружению:
  - Исправленный model_state.py (F.shape[0] = len(mesh.x)-1, центры из get_centers())
  - n_ghost = 3 для WENO5 (передаётся явно в get_smooth_convergence)
  - dtype = float64 (задан здесь; переопределяет configuration.py)
  - linspace(n_x+1) — n_x физических ячеек, не n_x-1
"""

import time
import numpy as np

from src.advection_solvers.WENO5RK3 import WENO5RK3
from src.advection_solvers.rk2 import SolverRK
from src.advection_solvers.godunov import SolverGodunov
from src.advection_solvers.kolgan import SolverKolgan
from src.thermodynamics.boundary_condition import PeriodicBoundaryCondition
from src.config.configuration import XI_LEFT, XI_RIGHT, TD_PR, TD_W, X_LEFT, X_RIGHT
from src.mesh import UnadaptableMesh

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from src.thermodynamics.model_properties import ModelProperties
from src.thermodynamics.model_state import ModelState
from src.thermodynamics.property_calculator import PropertyCalculator
from src.thermodynamics.shakhov_solver import ShakhovSolver
from src.utils.sod_exact import L2, L1, L_sup

from src.config.libloader import xp, cuda_is_available


# ── Параметры волны ───────────────────────────────────────────────────────────
#
# Конвенция температуры в коде: T_code = 2·T_thermo
# Максвелл: exp(-(ξ-u)²/T_code), поэтому p = n·T_code/2
# Скорость звука: c² = γ·p/ρ = γ·T_code/2

EPS     = 0.1          # амплитуда возмущения (0.1 — безопасно до t_max ≈ 1.7)
N0      = 1.0
U0      = 0.0
T0      = 1.0          # T_code фона
GAMMA   = 5.0 / 3.0
C_SOUND = np.sqrt(GAMMA * T0 / 2.0)   # ≈ 0.9129 — скорость звука


def _F_BEG_N(x):
    return N0 + EPS * xp.sin(2.0 * xp.pi * x)

def _F_BEG_U(x):
    # Правый бег: u = c · δn/n0  (изентропическая линеаризация)
    return EPS * C_SOUND * xp.sin(2.0 * xp.pi * x)

def _F_BEG_T(x):
    # δT_code/T0 = (γ-1)·δn/n0  (изентропическое условие)
    return T0 + (GAMMA - 1.0) * EPS * xp.sin(2.0 * xp.pi * x)


# ── Точное решение (уравнения Эйлера, ошибка O(Kn)) ─────────────────────────

def n_exact(x, t):
    return N0 + EPS * np.sin(2.0 * np.pi * (x - C_SOUND * t))

def u_exact(x, t):
    return EPS * C_SOUND * np.sin(2.0 * np.pi * (x - C_SOUND * t))

def T_exact(x, t):
    """Возвращает T в конвенции кода (T_code)."""
    return T0 + (GAMMA - 1.0) * EPS * np.sin(2.0 * np.pi * (x - C_SOUND * t))


# ── Основная функция ─────────────────────────────────────────────────────────

def get_smooth_convergence(advection_solver_cls, n_x_vals, n_xi_vals, n_ghost=3):
    """
    Параметры
    ---------
    advection_solver_cls : класс солвера (SolverGodunov, SolverRK, WENO5RK3, ...)
    n_x_vals             : список числа физических ячеек
    n_xi_vals            : список размеров сетки по скоростям
    n_ghost              : число ghost-ячеек (2 для Godunov/Kolgan/RK2, 3 для WENO5)

    Возвращает
    ----------
    errors : ndarray (len(n_x_vals), 3) — [L_inf, L1, L2] по плотности n
    times  : ndarray (len(n_x_vals),)   — времена расчёта в секундах
    """

    CFL   = 0.8
    t_max = 0.2
    TD_KN = 1e-6      # Kn << 1: газ в гидродинамическом пределе
    dtype = xp.float64  # float32 маскирует сходимость выше ~1-го порядка

    if len(n_x_vals) != len(n_xi_vals):
        raise ValueError('n_x_vals и n_xi_vals должны иметь одинаковую длину')

    calculation_times  = np.zeros(len(n_x_vals))
    calculation_errors = np.zeros((len(n_x_vals), 3))

    for i, (n_x, n_xi) in enumerate(zip(n_x_vals, n_xi_vals)):

        model_config = {
            'X_LEFT':  X_LEFT,  'X_RIGHT':  X_RIGHT,
            'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
            'F_BEG_N': _F_BEG_N,
            'F_BEG_U': _F_BEG_U,
            'F_BEG_T': _F_BEG_T,
            'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W,
            'dtype': dtype,
        }

        bc = PeriodicBoundaryCondition(n_ghost)

        # linspace(n_x + 1) → ровно n_x физических ячеек после добавления ghost-узлов
        mesh = UnadaptableMesh(
            xp.linspace(X_LEFT, X_RIGHT, n_x + 1, endpoint=True),
            bc.n_ghost
        )

        adv_solver = advection_solver_cls()
        properties = ModelProperties(model_config, mesh, bc)
        state      = ModelState(properties, model_config)
        solver     = ShakhovSolver(state, properties, adv_solver)

        t1 = time.time()
        solver.calculate(CFL, t_max)
        t2 = time.time()
        calculation_times[i] = t2 - t1

        # Физические центры ячеек (без ghost)
        x_centers = properties.mesh.get_centers()[bc.n_ghost:-bc.n_ghost]
        if cuda_is_available:
            x_centers = xp.asnumpy(x_centers)

        n_sol, u_sol, T_sol, _ = PropertyCalculator.get_solution_macros(state.F, properties)

        n_ref = n_exact(x_centers, t_max)

        l_inf = float(L_sup(x_centers, n_ref, n_sol))
        l1    = float(L1(x_centers, n_ref, n_sol))
        l2    = float(L2(x_centers, n_ref, n_sol))

        print(
            f'  n_x={n_x:4d}  t={t2-t1:6.2f}s'
            f'  L_inf={l_inf:.3e}  L1={l1:.3e}  L2={l2:.3e}'
        )
        calculation_errors[i] = [l_inf, l1, l2]

    return calculation_errors, calculation_times


# ── Вывод таблицы ────────────────────────────────────────────────────────────

def print_convergence_table(solver_name, n_x_vals, errors, times):
    labels = ['L_inf', 'L1   ', 'L2   ']
    w = 74
    print()
    print('=' * w)
    print(f'  {solver_name}  (бегущая волна, периодические ГУ, t={0.2})')
    print('-' * w)
    print(f'  {"n_x":>5}  {"L_inf":>10}  {"rate":>5}  {"L1":>10}  {"rate":>5}  {"L2":>10}  {"rate":>5}  {"t,s":>6}')
    print('-' * w)
    for i, nx in enumerate(n_x_vals):
        if i == 0:
            print(
                f'  {nx:>5}  {errors[i,0]:>10.3e}  {"—":>5}'
                f'  {errors[i,1]:>10.3e}  {"—":>5}'
                f'  {errors[i,2]:>10.3e}  {"—":>5}'
                f'  {times[i]:>6.2f}'
            )
        else:
            rates = [
                np.log2(errors[i-1, j] / errors[i, j]) if errors[i, j] > 0 else float('nan')
                for j in range(3)
            ]
            print(
                f'  {nx:>5}  {errors[i,0]:>10.3e}  {rates[0]:>5.2f}'
                f'  {errors[i,1]:>10.3e}  {rates[1]:>5.2f}'
                f'  {errors[i,2]:>10.3e}  {rates[2]:>5.2f}'
                f'  {times[i]:>6.2f}'
            )
    print('=' * w)


# ── График ───────────────────────────────────────────────────────────────────

def plot_solution(solver_name, n_x, n_xi, n_ghost, t_max=0.2):
    """Сравнение численного и точного решения для одного n_x."""
    dtype = xp.float64
    TD_KN = 1e-6

    model_config = {
        'X_LEFT': X_LEFT,  'X_RIGHT': X_RIGHT,
        'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
        'F_BEG_N': _F_BEG_N, 'F_BEG_U': _F_BEG_U, 'F_BEG_T': _F_BEG_T,
        'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W, 'dtype': dtype,
    }
    bc   = PeriodicBoundaryCondition(n_ghost)
    mesh = UnadaptableMesh(xp.linspace(X_LEFT, X_RIGHT, n_x+1), bc.n_ghost)
    adv  = eval(solver_name + '()')
    prop = ModelProperties(model_config, mesh, bc)
    st   = ModelState(prop, model_config)
    ShakhovSolver(st, prop, adv).calculate(0.8, t_max)

    x = prop.mesh.get_centers()[bc.n_ghost:-bc.n_ghost]
    if cuda_is_available:
        x = xp.asnumpy(x)
    n_sol, u_sol, T_sol, _ = PropertyCalculator.get_solution_macros(st.F, prop)

    x_fine = np.linspace(X_LEFT, X_RIGHT, 500)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f'{solver_name}, n_x={n_x}, n_xi={n_xi}, t={t_max}, Kn=1e-6')

    for ax, y_num, y_func, ylabel in zip(
        axs,
        [n_sol, u_sol, T_sol],
        [n_exact, u_exact, T_exact],
        ['n (density)', 'u (velocity)', 'T_code (temperature)']
    ):
        ax.plot(x_fine, y_func(x_fine, t_max), 'k--', lw=1.5, label='exact (Euler)')
        ax.scatter(x, y_num, s=10, color='blue', zorder=3)
        ax.plot(x, y_num, color='blue', lw=1, label=solver_name)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# ── Точка входа ──────────────────────────────────────────────────────────────

if __name__ == '__main__':

    N_X_VALS = [10, 20, 40, 80, 160]
    N_XI_VALS = [20] * len(N_X_VALS)

    # RK2 (n_ghost=2 достаточно для стенсила 2-го порядка)
    print('\n=== RK2 — гладкая волна ===')
    err_rk2, t_rk2 = get_smooth_convergence(SolverRK, N_X_VALS, N_XI_VALS, n_ghost=2)
    print_convergence_table('SolverRK', N_X_VALS, err_rk2, t_rk2)

    # WENO5+RK3 (n_ghost=3 — необходимо)
    print('\n=== WENO5+RK3 — гладкая волна ===')
    err_w5, t_w5 = get_smooth_convergence(WENO5RK3, N_X_VALS, N_XI_VALS, n_ghost=3)
    print_convergence_table('WENO5RK3', N_X_VALS, err_w5, t_w5)

    # Сравнительный график: ошибки vs n_x
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title('L2-сходимость на гладком решении')
    h = 1.0 / np.array(N_X_VALS)
    ax.loglog(N_X_VALS, err_rk2[:, 2],  'b-o', label='RK2 (L2)')
    ax.loglog(N_X_VALS, err_w5[:, 2],   'r-s', label='WENO5+RK3 (L2)')
    # Эталонные линии
    ax.loglog(N_X_VALS, 0.5 * h**2, 'b--', lw=0.8, label='O(h²)')
    ax.loglog(N_X_VALS, 2.0 * h**3, 'r--', lw=0.8, label='O(h³)')
    ax.set_xlabel('n_x')
    ax.set_ylabel('L2 error')
    ax.legend()
    ax.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.show()

    # Детальный график решения для крупнейшей сетки
    plot_solution('WENO5RK3', n_x=80, n_xi=20, n_ghost=3)