"""
test_conv_free_molecular_smooth.py
-----------------------------------
Тест сходимости на гладком решении — свободномолекулярный режим (Kn = 1e10).

Физика:
  При Kn → ∞ уравнение Больцмана вырождается в уравнение свободного переноса:
      ∂f/∂t + ξ · ∂f/∂x = 0.
  Точное решение: f(x, ξ, t) = f0(x − ξ·t, ξ).

  Начальное условие — максвелловское распределение с синусоидальным
  возмущением плотности на периодической области [0, 1]:
      f0(x, ξ) = [N0 + EPS·sin(2π·x)] / sqrt(π·T0) · exp(−ξ²/T0).

  Точная плотность (характеристическая функция Гауссиана):
      n_exact(x, t) = N0 + EPS · sin(2π·x) · exp(−π²·T0·t²).
  Амплитуда затухает из-за фазового перемешивания: молекулы с разными ξ
  сдвигаются на разные расстояния, разрушая пространственную структуру.

Почему это лучше акустического теста (test_conv_smooth.py):
  - Точное решение известно аналитически на уровне кинетики, а не только
    в гидродинамическом пределе.
  - Нет нелинейного пола O(ε²) от разницы линейной и нелинейной Эйлера.
  - Нет ограничения по порядку от оператора столкновений (BGK отсутствует
    при Kn → ∞): WENO5 может показать свой истинный пространственный порядок.
  - При τ ~ h и CFL = const временна́я ошибка WENO5+RK3 — O(h³),
    пространственная — O(h⁵). На грубых сетках видна O(h⁵), на тонких — O(h³).

Ожидаемые порядки по L2:
  Godunov            ~ 1
  Kolgan / RK2       ~ 2
  WENO5 + SSP-RK3    ~ 5 (грубые сетки) → 3 (тонкие, τ-ограничение)

Граничные условия: периодические.
Параметры: N_xi = 20, CFL = 0.8, t_max = 0.1.
"""

import time
import numpy as np

from src.advection_solvers.WENO5RK3_IMEX import WENO5RK3
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


# ── Параметры ─────────────────────────────────────────────────────────────────

EPS   = 1e-3   # амплитуда (линейный режим: нет нелинейного пола O(ε²))
N0    = 1.0
U0    = 0.0
T0    = 1.0    # T_code (Максвелл ~ exp(−ξ²/T_code))
TD_KN = 1e10   # свободномолекулярный режим: столкновения отсутствуют


def _F_BEG_N(x):
    return N0 + EPS * xp.sin(2.0 * xp.pi * x)

def _F_BEG_U(x):
    return xp.zeros_like(x) + U0

def _F_BEG_T(x):
    return xp.zeros_like(x) + T0


# ── Точное решение ────────────────────────────────────────────────────────────
#
# f(x, ξ, t) = f0(x − ξ·t, ξ)
#            = [N0 + EPS·sin(2π·(x−ξt))] / sqrt(π·T0) · exp(−ξ²/T0)
#
# n(x, t) = ∫ f dξ
#          = N0 + EPS·sin(2π·x) · ∫ cos(2π·ξt) · G(ξ) dξ
#                                  − EPS·cos(2π·x) · ∫ sin(2π·ξt) · G(ξ) dξ
#
# Для симметричного G(ξ) = exp(−ξ²/T0)/sqrt(π·T0): ∫ sin(·) G dξ = 0 (нечётный подынтегральный)
# ∫ cos(2π·ξt) G dξ = Re[∫ exp(2πi·ξt) G dξ] = exp(−π²·T0·t²)  (хар. функция Гауссиана)
#
# Итог: n(x, t) = N0 + EPS · sin(2π·x) · exp(−π²·T0·t²)

def n_exact(x, t):
    """Точная плотность: затухающая синусоида."""
    decay = np.exp(-np.pi**2 * T0 * t**2)
    return N0 + EPS * np.sin(2.0 * np.pi * x) * decay

def n_exact_decay(t):
    """Коэффициент затухания амплитуды."""
    return np.exp(-np.pi**2 * T0 * t**2)


# ── Основная функция ──────────────────────────────────────────────────────────

def get_free_molecular_smooth_convergence(
    advection_solver_cls, n_x_vals, n_xi_vals, n_ghost=3, CFL=0.8
):
    """
    Параметры
    ---------
    advection_solver_cls : класс солвера (SolverGodunov, SolverRK, WENO5RK3, …)
    n_x_vals             : список числа физических ячеек по x
    n_xi_vals            : список размеров сетки по ξ
    n_ghost              : число ghost-ячеек (2 для Godunov/RK2, 3 для WENO5)

    Возвращает
    ----------
    errors : ndarray (len(n_x_vals), 3) — [L_inf, L1, L2] по плотности n
    times  : ndarray (len(n_x_vals),)   — времена расчёта в секундах
    """

    t_max = 0.1   # decay ≈ 0.37 — волна заметно затухла, но разрешима
    dtype = xp.float64

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

        bc   = PeriodicBoundaryCondition(n_ghost)
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

        x_centers = properties.mesh.get_centers()[bc.n_ghost:-bc.n_ghost]
        if cuda_is_available:
            x_centers = xp.asnumpy(x_centers)

        n_sol, _, _, _ = PropertyCalculator.get_solution_macros(state.F, properties)

        n_ref = n_exact(x_centers, t_max)

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
"""
def print_convergence_table(solver_name, n_x_vals, errors, times, t_max=0.1):
    w = 74
    print()
    print('=' * w)
    print(f'  {solver_name}  (свободномолекулярный режим, Kn=1e10, t={t_max})')
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
    """
def print_convergence_table(solver_name, n_x_vals, errors, times, t_max=0.1):
    w = 74
    print()
    print('=' * w)
    print(f'  {solver_name}  (свободномолекулярный режим, Kn=1e10, t={t_max})')
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
            # порядок при произвольном отношении сеток:
            #   p = ln(e_{i-1}/e_i) / ln(N_i / N_{i-1})
            ratio = n_x_vals[i] / n_x_vals[i - 1]
            log_ratio = np.log(ratio)
            rates = [
                np.log(errors[i - 1, j] / errors[i, j]) / log_ratio
                if (errors[i, j] > 0 and errors[i - 1, j] > 0 and ratio != 1)
                else float('nan')
                for j in range(3)
            ]
            print(
                f'  {nx:>5}  {errors[i,0]:>10.3e}  {rates[0]:>5.2f}'
                f'  {errors[i,1]:>10.3e}  {rates[1]:>5.2f}'
                f'  {errors[i,2]:>10.3e}  {rates[2]:>5.2f}'
                f'  {times[i]:>6.2f}'
            )
    print('=' * w)


# ── График ─────────────────────────────────────────────────────────────────────

def plot_profile(solver_name_cls_ghost, n_x, n_xi, t_max=0.1):
    """
    Детальный график n(x): численное vs точное.
    solver_name_cls_ghost — тройка (имя, класс, n_ghost).
    """
    dtype = xp.float64
    name, cls, n_ghost = solver_name_cls_ghost

    model_config = {
        'X_LEFT': X_LEFT,  'X_RIGHT': X_RIGHT,
        'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
        'F_BEG_N': _F_BEG_N, 'F_BEG_U': _F_BEG_U, 'F_BEG_T': _F_BEG_T,
        'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W, 'dtype': dtype,
    }
    bc   = PeriodicBoundaryCondition(n_ghost)
    mesh = UnadaptableMesh(xp.linspace(X_LEFT, X_RIGHT, n_x + 1), bc.n_ghost)
    prop = ModelProperties(model_config, mesh, bc)
    st   = ModelState(prop, model_config)
    ShakhovSolver(st, prop, cls()).calculate(0.8, t_max)

    x = prop.mesh.get_centers()[bc.n_ghost:-bc.n_ghost]
    if cuda_is_available:
        x = xp.asnumpy(x)
    n_sol, _, _, _ = PropertyCalculator.get_solution_macros(st.F, prop)

    x_fine = np.linspace(X_LEFT, X_RIGHT, 2000)
    decay  = n_exact_decay(t_max)

    FSIZE = 16
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_fine, n_exact(x_fine, t_max), 'k-', lw=1.5,
            label=f'Точное решение')
    ax.plot(x, n_sol, 'o-', ms=3, lw=1, color='tab:orange', label=name)
    ax.set_xlabel('x', fontsize=FSIZE)
    ax.set_ylabel('n(x)', fontsize=FSIZE)
    ax.set_title(
        f'{name},  $N_x={n_x}$,  $N_\\xi={n_xi}$,  $t={t_max}$,  Kn=10¹⁰',
        fontsize=FSIZE,
    )
    ax.tick_params(axis='both', labelsize=FSIZE - 2)
    ax.legend(fontsize=FSIZE - 2)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ── Точка входа ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    N_X_VALS  = [10, 20, 40, 80, 160, 320, 640]
    N_XI_VALS = [20] * len(N_X_VALS)

    print(f'\nt_max=0.1:  decay = {n_exact_decay(0.1):.4f}  '
          f'(амплитуда sin уменьшилась до {n_exact_decay(0.1)*100:.1f}%)')

    # ── Godunov ──────────────────────────────────────────────────────────────
    print('\n=== Godunov ===')
    err_god, t_god = get_free_molecular_smooth_convergence(
        SolverGodunov, N_X_VALS, N_XI_VALS, n_ghost=2
    )
    print_convergence_table('SolverGodunov', N_X_VALS, err_god, t_god)

        # ── Kolgan ───────────────────────────────────────────────────────────────

    print('\n=== Kolgan ===')
    #CFL   = 0.45
    err_klg, t_klg = get_free_molecular_smooth_convergence(
        SolverKolgan, N_X_VALS, N_XI_VALS, n_ghost=2, CFL=0.5
    )
    print_convergence_table('SolverKolgan', N_X_VALS, err_klg, t_klg)
    #CFL   = 0.8
    # ── RK2 ──────────────────────────────────────────────────────────────────
    print('\n=== RK2 ===')
    err_rk2, t_rk2 = get_free_molecular_smooth_convergence(
        SolverRK, N_X_VALS, N_XI_VALS, n_ghost=3, CFL=0.5
    )
    print_convergence_table('SolverRK', N_X_VALS, err_rk2, t_rk2)

    # ── WENO5+RK3 ────────────────────────────────────────────────────────────
    print('\n=== WENO5+RK3 ===')
    err_w5, t_w5 = get_free_molecular_smooth_convergence(
        WENO5RK3, N_X_VALS, N_XI_VALS, n_ghost=3
    )
    print_convergence_table('WENO5RK3', N_X_VALS, err_w5, t_w5)

    # ── L2-график ────────────────────────────────────────────────────────────
    FSIZE = 16   # базовый размер шрифта для графиков
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title('L2-сходимость, свободномолекулярный режим, Kn=1e10',
                 fontsize=FSIZE)
    h = 1.0 / np.array(N_X_VALS)
    ax.loglog(N_X_VALS, err_god[:, 2], 'g-^', label='Godunov')
    ax.loglog(N_X_VALS, err_klg[:, 2], 'm-v', label='Kolgan')
    ax.loglog(N_X_VALS, err_rk2[:, 2], 'b-o', label='RK2')
    ax.loglog(N_X_VALS, err_w5[:, 2],  'r-s', label='WENO5+RK3')
    # эталонные наклоны
    #ax.loglog(N_X_VALS, err_god[0, 2] * (h / h[0])**1, 'g--', lw=0.8, label='O(h¹)')
    #ax.loglog(N_X_VALS, err_rk2[0, 2] * (h / h[0])**2, 'b--', lw=0.8, label='O(h²)')
    #ax.loglog(N_X_VALS, err_w5[0,  2] * (h / h[0])**5, 'r--', lw=0.8, label='O(h⁵)')
    #ax.loglog(N_X_VALS, err_w5[1,  2] * (h / h[1])**3, 'r:',  lw=0.8, label='O(h³)')
    ax.set_xlabel('$N_x$', fontsize=FSIZE)
    ax.set_ylabel('$\\varepsilon_{L_2}$', fontsize=FSIZE)
    ax.tick_params(axis='both', labelsize=FSIZE - 2)
    ax.legend(fontsize=FSIZE - 2)
    ax.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.show()

    # ── Профиль на плотной сетке ─────────────────────────────────────────────
    #plot_profile(('WENO5+RK3', WENO5RK3, 3), n_x=160, n_xi=20)
    #plot_profile(('Kolgan', SolverKolgan, 3), n_x=160, n_xi=20)