"""
test_conv_Sod.py
----------------
Тест сходимости на задаче Сода — гидродинамический предел (Kn = 1e-5).

Физика:
  При Kn → 0 уравнение Больцмана сводится к уравнениям Эйлера.
  Задача Сода — стандартный тест Римана с начальными условиями:

      x ≤ 0.5:  ρ = 1.0,   u = 0,  T = 1.0
      x > 0.5:  ρ = 0.125, u = 0,  T = 0.8

  Точное решение уравнений Эйлера (γ = 5/3) содержит три волны:
    - волну разрежения (слева),
    - контактный разрыв (в центре),
    - ударную волну (справа).

  Опорное решение берётся из euler_exact() — аналитическое решение задачи
  Римана для идеального газа с γ = 5/3.

Ожидаемые порядки по L1 для плотности:
  Godunov            ~ 1
  Kolgan             ~ 1–2 (ограничитель снижает порядок на разрывах)
  RK2                ~ 1–2 (аналогично Kolgan)

  Из-за присутствия разрывов порядок выше первого на равномерных сетках
  недостижим в нормах L∞ и L2; L1 обычно даёт наиболее репрезентативную
  картину. Ожидаемый порядок L1 ≈ 1 (Godunov) и ≈ 1–2 (схемы 2-го порядка).

Параметры:
  N_xi = 20, CFL = 0.8, t_max = 0.2, Kn = 1e-5.
  Граничные условия: нулевой градиент (ZeroGrad).
"""

import time
import numpy as np

from src.advection_solvers.WENO5RK3_IMEX import WENO5RK3
from src.advection_solvers.rk2 import SolverRK
from src.advection_solvers.godunov import SolverGodunov
from src.advection_solvers.kolgan import SolverKolgan
from src.thermodynamics.boundary_condition import ZeroGradBoundaryCondition
from src.config.configuration import XI_LEFT, XI_RIGHT, TD_PR, TD_W, X_LEFT, X_RIGHT
from src.mesh import UnadaptableMesh

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from src.thermodynamics.model_properties import ModelProperties
from src.thermodynamics.model_state import ModelState
from src.thermodynamics.property_calculator import PropertyCalculator
from src.thermodynamics.shakhov_solver import ShakhovSolver
from src.utils.sod_exact import euler_exact, L2, L1, L_sup

from src.config.libloader import xp, cuda_is_available


# ── Параметры ─────────────────────────────────────────────────────────────────

TD_KN = 1e-5   # гидродинамический предел: частые столкновения → Эйлер


def F_BEG_N(x):
    return xp.where(x <= 0.5, 1.0, 0.125)

def F_BEG_U(x):
    return xp.zeros_like(x)

def F_BEG_T(x):
    return xp.where(x <= 0.5, 1.0, 0.8)


# ── Точное решение ────────────────────────────────────────────────────────────
#
# Точное решение задачи Римана для уравнений Эйлера с γ = 5/3:
#   euler_exact(x, x0, rhoL, rhoR, uL, uR, pL, pR, t, gamma)
#
# Начальные давления: pL = T_L * rho_L / 2 = 1/2, pR = 0.8 * 0.125 / 2 = 0.05.
# (Фактор 1/2 соответствует нормировке в src.utils.sod_exact.)

def get_exact(x, t_max):
    """Возвращает (n_exact, u_exact, T_exact) на сетке x в момент t_max."""
    return euler_exact(
        x, 0.5,
        1.0, 0.125,   # rhoL, rhoR
        0.0, 0.0,     # uL, uR
        1.0 / 2.0,    # pL
        0.8 / 2.0,    # pR
        t_max,
        gamma=5.0 / 3.0,
    )


# ── Основная функция ──────────────────────────────────────────────────────────

def get_Sod_convergence(
    advection_solver_cls, n_x_vals, n_xi_vals, n_ghost=2
):
    """
    Параметры
    ---------
    advection_solver_cls : класс солвера (SolverGodunov, SolverRK, WENO5RK3, …)
    n_x_vals             : список числа физических ячеек по x
    n_xi_vals            : список размеров сетки по ξ
    n_ghost              : число ghost-ячеек (2 для Godunov/Kolgan/RK2, 3 для WENO5)

    Возвращает
    ----------
    errors : ndarray (len(n_x_vals), 3) — [L_inf, L1, L2] по плотности n
    times  : ndarray (len(n_x_vals),)   — времена расчёта в секундах
    """

    CFL   = 0.8
    t_max = 0.2
    dtype = xp.float32

    if len(n_x_vals) != len(n_xi_vals):
        raise ValueError('n_x_vals и n_xi_vals должны иметь одинаковую длину')

    calculation_times  = np.zeros(len(n_x_vals))
    calculation_errors = np.zeros((len(n_x_vals), 3))

    for i, (n_x, n_xi) in enumerate(zip(n_x_vals, n_xi_vals)):

        model_config = {
            'X_LEFT':  X_LEFT,  'X_RIGHT':  X_RIGHT,
            'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
            'F_BEG_N': F_BEG_N,
            'F_BEG_U': F_BEG_U,
            'F_BEG_T': F_BEG_T,
            'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W,
            'dtype': dtype,
        }

        bc   = ZeroGradBoundaryCondition(n_ghost)
        mesh = UnadaptableMesh(
            xp.linspace(X_LEFT, X_RIGHT, n_x + 1, endpoint=True),
            bc.n_ghost,
        )

        adv_solver = advection_solver_cls()
        properties = ModelProperties(model_config, mesh, bc)
        state      = ModelState(properties, model_config)
        solver     = ShakhovSolver(state, properties, adv_solver)

        t1 = time.time()
        solver.calculate(CFL, t_max)
        t2 = time.time()
        calculation_times[i] = t2 - t1

        x_centers = properties.mesh.get_centers()[bc.n_ghost:-bc.n_ghost]
        if cuda_is_available:
            x_centers = xp.asnumpy(x_centers)

        n_sol, _, _, _ = PropertyCalculator.get_solution_macros(state.F, properties)

        n_ref, _, _ = get_exact(x_centers, t_max)

        l_inf = float(L_sup(x_centers, n_ref, n_sol))
        l1    = float(L1(x_centers, n_ref, n_sol))
        l2    = float(L2(x_centers, n_ref, n_sol))

        print(
            f'  n_x={n_x:4d}  n_xi={n_xi:3d}  t={t2-t1:6.2f}s'
            f'  L_inf={l_inf:.3e}  L1={l1:.3e}  L2={l2:.3e}'
        )
        calculation_errors[i] = [l_inf, l1, l2]

    return calculation_errors, calculation_times


# ── Вывод таблицы ─────────────────────────────────────────────────────────────

def print_convergence_table(solver_name, n_x_vals, errors, times, t_max=0.2):
    w = 74
    print()
    print('=' * w)
    print(f'  {solver_name}  (задача Сода, Kn=1e-5, t={t_max})')
    print('-' * w)
    print(f'  {"n_x":>5}  {"L_inf":>10}  {"rate":>5}  '
          f'{"L1":>10}  {"rate":>5}  {"L2":>10}  {"rate":>5}  {"t,s":>6}')
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


# ── График профиля ─────────────────────────────────────────────────────────────

def plot_profile(solver_name_cls_ghost, n_x, n_xi, t_max=0.2):
    """
    Детальный график n(x): численное vs точное (Эйлер).
    solver_name_cls_ghost — тройка (имя, класс, n_ghost).
    """
    dtype = xp.float32
    name, cls, n_ghost = solver_name_cls_ghost

    model_config = {
        'X_LEFT':  X_LEFT,  'X_RIGHT':  X_RIGHT,
        'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
        'F_BEG_N': F_BEG_N, 'F_BEG_U': F_BEG_U, 'F_BEG_T': F_BEG_T,
        'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W, 'dtype': dtype,
    }
    bc   = ZeroGradBoundaryCondition(n_ghost)
    mesh = UnadaptableMesh(xp.linspace(X_LEFT, X_RIGHT, n_x + 1), bc.n_ghost)
    prop = ModelProperties(model_config, mesh, bc)
    st   = ModelState(prop, model_config)
    ShakhovSolver(st, prop, cls()).calculate(0.8, t_max)

    x = prop.mesh.get_centers()[bc.n_ghost:-bc.n_ghost]
    if cuda_is_available:
        x = xp.asnumpy(x)
    n_sol, _, _, _ = PropertyCalculator.get_solution_macros(st.F, prop)

    x_fine = np.linspace(X_LEFT, X_RIGHT, 2000)
    n_ref, _, _ = get_exact(x_fine, t_max)

    FSIZE = 16
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_fine, n_ref, 'k-', lw=1.5, label='Точное решение (Эйлер)')
    ax.plot(x, n_sol, 'o-', ms=3, lw=1, color='tab:orange', label=name)
    ax.set_xlabel('x', fontsize=FSIZE)
    ax.set_ylabel('n(x)', fontsize=FSIZE)
    ax.set_title(
        f'{name},  $N_x={n_x}$,  $N_\\xi={n_xi}$,  $t={t_max}$,  Kn=10⁻⁵',
        fontsize=FSIZE,
    )
    ax.tick_params(axis='both', labelsize=FSIZE - 2)
    ax.legend(fontsize=FSIZE - 2)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ── Вспомогательная функция: один прогон ──────────────────────────────────────

def run_solver(advection_solver_cls, n_x, n_xi, n_ghost=2, t_max=0.2, cfl=0.8):
    """
    Запускает один расчёт и возвращает (x_centers, n_sol).
    Используется в plot_all_schemes для сбора профилей без повторения кода.
    """
    dtype = xp.float32
    model_config = {
        'X_LEFT':  X_LEFT,  'X_RIGHT':  X_RIGHT,
        'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
        'F_BEG_N': F_BEG_N, 'F_BEG_U': F_BEG_U, 'F_BEG_T': F_BEG_T,
        'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W, 'dtype': dtype,
    }
    bc   = ZeroGradBoundaryCondition(n_ghost)
    mesh = UnadaptableMesh(xp.linspace(X_LEFT, X_RIGHT, n_x + 1), bc.n_ghost)
    prop = ModelProperties(model_config, mesh, bc)
    st   = ModelState(prop, model_config)
    ShakhovSolver(st, prop, advection_solver_cls()).calculate(cfl, t_max)

    x = prop.mesh.get_centers()[bc.n_ghost:-bc.n_ghost]
    if cuda_is_available:
        x = xp.asnumpy(x)
    n_sol, _, _, _ = PropertyCalculator.get_solution_macros(st.F, prop)
    return x, n_sol


# ── Сравнительный график всех схем ────────────────────────────────────────────

def plot_all_schemes(n_x, n_xi, t_max=0.2):
    """
    Прогоняет все четыре схемы на одной сетке (n_x × n_xi) и строит
    сравнительный график плотности n(x).

    Точное решение — толстая чёрная линия.
    Каждая схема — тонкая линия с маркерами, цвета совпадают с графиком сходимости.
    Позволяет визуально оценить монотонность (отсутствие/наличие осцилляций).

    Параметры
    ---------
    n_x   : число физических ячеек по x
    n_xi  : число ячеек по ξ
    t_max : момент времени (по умолчанию 0.2)
    """
    FSIZE = 16

    schemes = [
        ('Godunov',   SolverGodunov, 2, 'g-^'),
        ('Kolgan',    SolverKolgan,  2, 'm-v'),
        ('RK2',       SolverRK,      2, 'b-o'),
        ('WENO5+RK3', WENO5RK3,      3, 'r-s'),
    ]

    x_fine = np.linspace(X_LEFT, X_RIGHT, 2000)
    n_ref, _, _ = get_exact(x_fine, t_max)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_fine, n_ref, 'k-', lw=2, zorder=5, label='Точное решение (Эйлер)')

    for name, cls, n_ghost, fmt in schemes:
        print(f'  Запуск {name}  (n_x={n_x}, n_xi={n_xi}) ...')
        x, n_sol = run_solver(cls, n_x, n_xi, n_ghost=n_ghost, t_max=t_max)
        ax.plot(x, n_sol, fmt, ms=3, lw=1, label=name)

    ax.set_xlabel('x', fontsize=FSIZE)
    ax.set_ylabel('n(x)', fontsize=FSIZE)
    ax.set_title(
        f'Сравнение схем, задача Сода,  '
        f'$N_x={n_x}$,  $N_\\xi={n_xi}$,  $t={t_max}$,  Kn=10⁻⁵',
        fontsize=FSIZE,
    )
    ax.tick_params(axis='both', labelsize=FSIZE - 2)
    ax.legend(fontsize=FSIZE - 2)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ── Точка входа ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    N_X_VALS  = [10, 20, 40, 80, 160, 320]
    N_XI_VALS = [20] * len(N_X_VALS)

    # ── Godunov ──────────────────────────────────────────────────────────────
    print('\n=== Godunov ===')
    err_god, t_god = get_Sod_convergence(
        SolverGodunov, N_X_VALS, N_XI_VALS, n_ghost=2
    )
    print_convergence_table('SolverGodunov', N_X_VALS, err_god, t_god)

    # ── Kolgan ───────────────────────────────────────────────────────────────
    print('\n=== Kolgan ===')
    err_klg, t_klg = get_Sod_convergence(
        SolverKolgan, N_X_VALS, N_XI_VALS, n_ghost=2
    )
    print_convergence_table('SolverKolgan', N_X_VALS, err_klg, t_klg)

    # ── RK2 ──────────────────────────────────────────────────────────────────
    print('\n=== RK2 ===')
    err_rk2, t_rk2 = get_Sod_convergence(
        SolverRK, N_X_VALS, N_XI_VALS, n_ghost=2
    )
    print_convergence_table('SolverRK', N_X_VALS, err_rk2, t_rk2)

    # ── WENO5+RK3 ────────────────────────────────────────────────────────────
    print('\n=== WENO5+RK3 ===')
    err_w5, t_w5 = get_Sod_convergence(
        WENO5RK3, N_X_VALS, N_XI_VALS, n_ghost=3
    )
    print_convergence_table('WENO5RK3', N_X_VALS, err_w5, t_w5)

    # ── L1-график сходимости ─────────────────────────────────────────────────
    # Для задач с разрывами L1 — наиболее репрезентативная норма.
    FSIZE = 16
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title('L1-сходимость, задача Сода, Kn=1e-5', fontsize=FSIZE)
    h = 1.0 / np.array(N_X_VALS)
    ax.loglog(N_X_VALS, err_god[:, 1], 'g-^', label='Godunov')
    ax.loglog(N_X_VALS, err_klg[:, 1], 'm-v', label='Kolgan')
    ax.loglog(N_X_VALS, err_rk2[:, 1], 'b-o', label='RK2')
    ax.loglog(N_X_VALS, err_w5[:, 1],  'r-s', label='WENO5+RK3')
    # эталонные наклоны
    ax.loglog(N_X_VALS, err_god[0, 1] * (h / h[0])**1, 'g--', lw=0.8, label='O(h¹)')
    ax.loglog(N_X_VALS, err_rk2[0, 1] * (h / h[0])**2, 'b--', lw=0.8, label='O(h²)')
    ax.set_xlabel('$N_x$', fontsize=FSIZE)
    ax.set_ylabel('$\\varepsilon_{L_1}$', fontsize=FSIZE)
    ax.tick_params(axis='both', labelsize=FSIZE - 2)
    ax.legend(fontsize=FSIZE - 2)
    ax.grid(True, which='both', ls=':')
    plt.tight_layout()
    #plt.show()
    plt.savefig('./conv.png')

    # ── Профиль на плотной сетке (одна схема) ───────────────────────────────
    plot_profile(('WENO5+RK3', WENO5RK3, 3), n_x=160, n_xi=40)

    # ── Сравнение всех схем на одном графике ────────────────────────────────
    # Грубая сетка: осцилляции у схем без ограничителя хорошо видны.
    plot_all_schemes(n_x=100, n_xi=40)