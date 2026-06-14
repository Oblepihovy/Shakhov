"""
test_conv_free_molecular.py
---------------------------
Тест сходимости на гладком решении — свободномолекулярный режим (Kn = 1e10).

Физика:
  При Kn → ∞ столкновения отсутствуют. Уравнение Больцмана вырождается
  в уравнение свободного переноса:
      ∂f/∂t + ξ · ∂f/∂x = 0
  Точное решение: f(x, ξ, t) = f0(x - ξ·t, ξ)  — сдвиг по x на ξ·t.

  Начальное условие — максвелловское распределение с синусоидальным
  возмущением плотности (изентропически согласованное):
      f0(x, ξ) = n(x) / sqrt(π·T0) · exp(-(ξ - u0)²/T0)
  где n(x) = N0 + EPS·sin(2π·x).

  Макро-переменные из точного f(x, ξ, t):
      n_exact(x, t) = ∫ f0(x - ξ·t, ξ) dξ
  Аналитически: n_exact(x,t) = N0 + EPS · Re[exp(2πi·x) · Φ(t)]
  где Φ(t) = exp(-π²·T0·t²) — затухание из-за фазового перемешивания
  (разные ξ сдвигаются по-разному → «size mixing» уничтожает структуру).

  Точное n(x,t) вычисляется численным интегрированием по ξ.

Ожидаемые порядки:
  Godunov            ~1
  Kolgan / RK2       ~2
  WENO5 + SSP-RK3    ~5  (ошибка коллизий = 0, доминирует пространственная схема)

Граничные условия: периодические.
"""

import time
import numpy as np
from scipy import integrate

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

EPS   = 1e-3   # амплитуда возмущения (линейный режим — затухание аналитично)
N0    = 1.0
U0    = 0.0
T0    = 1.0    # T_code фона (конвенция кода: Maxwell ~ exp(-(ξ-u)²/T_code))
TD_KN = 1e10   # свободномолекулярный режим: столкновения отсутствуют


def _F_BEG_N(x):
    return N0 + EPS * xp.sin(2.0 * xp.pi * x)

def _F_BEG_U(x):
    return xp.zeros_like(x) + U0   # u0 = 0: нет потока скорости

def _F_BEG_T(x):
    return xp.zeros_like(x) + T0   # температура однородная


# ── Точное решение ────────────────────────────────────────────────────────────
#
# f0(x, ξ) = [N0 + EPS·sin(2π·x)] / sqrt(π·T0) · exp(-ξ²/T0)
#
# f(x, ξ, t) = f0(x - ξ·t, ξ)
#
# n(x, t) = ∫_{-∞}^{+∞} f(x, ξ, t) dξ
#          = ∫ [N0 + EPS·sin(2π·(x - ξ·t))] / sqrt(π·T0) · exp(-ξ²/T0) dξ
#          = N0  +  EPS · sin(2π·x) · exp(-π²·T0·t²)
#
# Последнее — характеристическая функция Гауссова распределения:
#   ∫ sin(2π·(x-ξt)) · G(ξ) dξ = sin(2π·x) · Re[exp(-2πi·t·<ξ>_G)]
# Для G ~ exp(-ξ²/T0)/sqrt(π·T0):  <exp(ikξ)> = exp(-k²·T0/4)
# При k = 2π·t:  экспонента = exp(-π²·T0·t²)   ✓

def n_exact(x, t):
    """
    Точная плотность в свободномолекулярном режиме.
    Аналитическое выражение: затухающая синусоида.
    """
    decay = np.exp(-np.pi**2 * T0 * t**2)
    return N0 + EPS * np.sin(2.0 * np.pi * x) * decay

def u_exact(x, t):
    """
    Точная скорость u = (1/n) ∫ ξ·f dξ.
    При малой амплитуде EPS → 0 можно линеаризовать.
    Для однородного T0 и u0=0: <ξ> ≈ EPS·C·sin(2π·(x-...))·decay/n
    Используем численное интегрирование для честности.
    """
    return _macro_numerical(x, t, moment=1) / np.maximum(n_exact(x, t), 1e-30)

def T_exact(x, t):
    """
    T_code = (2/n) ∫ (ξ - u)²·f dξ  — конвенция кода.
    """
    n  = n_exact(x, t)
    u  = u_exact(x, t)
    m2 = _macro_numerical(x, t, moment=2)
    return (m2 - n * u**2) / np.maximum(n, 1e-30)


def _macro_numerical(x_arr, t, moment=0, n_quad=200):
    """
    Численное интегрирование по ξ для момента <ξ^moment · f>.
    Используется только для T_exact и u_exact (справочно).
    """
    xi_range = np.linspace(XI_LEFT, XI_RIGHT, n_quad)
    dxi = xi_range[1] - xi_range[0]
    result = np.zeros_like(x_arr, dtype=float)
    for j, xi in enumerate(xi_range):
        x_shifted = x_arr - xi * t
        f0 = (N0 + EPS * np.sin(2.0 * np.pi * x_shifted)) \
             / np.sqrt(np.pi * T0) * np.exp(-xi**2 / T0)
        result += xi**moment * f0 * dxi
    return result


# ── Основная функция ──────────────────────────────────────────────────────────

def get_free_molecular_convergence(advection_solver_cls, n_x_vals, n_xi_vals, n_ghost=3):
    """
    Параметры
    ---------
    advection_solver_cls : класс солвера
    n_x_vals             : список числа физических ячеек по x
    n_xi_vals            : список размеров сетки по ξ
    n_ghost              : число ghost-ячеек (2 для Godunov/Kolgan/RK2, 3 для WENO5)

    Возвращает
    ----------
    errors : ndarray (len(n_x_vals), 3) — [L_inf, L1, L2] по плотности n
    times  : ndarray (len(n_x_vals),)   — времена расчёта в секундах

    Примечание о t_max
    ------------------
    Берём t_max = 0.1 — к этому моменту decay = exp(-π²·t²) ≈ 0.37,
    т.е. волна заметно затухла, но ещё разрешима на сетке.
    """

    CFL   = 0.8
    t_max = 0.1
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
        solver.calculate(CFL, t_max, print_each=10)
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
            f'  n_x={n_x:4d}  n_xi={n_xi:3d}  t={t2-t1:6.2f}s'
            f'  L_inf={l_inf:.3e}  L1={l1:.3e}  L2={l2:.3e}'
        )
        calculation_errors[i] = [l_inf, l1, l2]

    return calculation_errors, calculation_times


# ── Вывод таблицы ─────────────────────────────────────────────────────────────

def print_convergence_table(solver_name, n_x_vals, errors, times, t_max=0.1):
    w = 74
    print()
    print('=' * w)
    print(f'  {solver_name}  (свободномолекулярный режим, Kn=1e10, t={t_max})')
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


# ── Детальный график решения ──────────────────────────────────────────────────

def plot_solution(solver_name, n_x, n_xi, n_ghost, t_max=0.1):
    """Сравнение численного и точного решения (n, u, T) для одного n_x."""
    dtype = xp.float64

    model_config = {
        'X_LEFT': X_LEFT,  'X_RIGHT': X_RIGHT,
        'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
        'F_BEG_N': _F_BEG_N, 'F_BEG_U': _F_BEG_U, 'F_BEG_T': _F_BEG_T,
        'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W, 'dtype': dtype,
    }
    bc   = PeriodicBoundaryCondition(n_ghost)
    mesh = UnadaptableMesh(xp.linspace(X_LEFT, X_RIGHT, n_x + 1), bc.n_ghost)

    solver_map = {
        'SolverGodunov': SolverGodunov,
        'SolverRK':      SolverRK,
        'SolverKolgan':  SolverKolgan,
        'WENO5RK3':      WENO5RK3,
    }
    adv  = solver_map[solver_name]()
    prop = ModelProperties(model_config, mesh, bc)
    st   = ModelState(prop, model_config)
    ShakhovSolver(st, prop, adv).calculate(0.8, t_max)

    x = prop.mesh.get_centers()[bc.n_ghost:-bc.n_ghost]
    if cuda_is_available:
        x = xp.asnumpy(x)
    n_sol, u_sol, T_sol, _ = PropertyCalculator.get_solution_macros(st.F, prop)

    x_fine = np.linspace(X_LEFT, X_RIGHT, 500)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f'{solver_name}, n_x={n_x}, n_xi={n_xi}, t={t_max}, Kn=1e10\n'
        f'(свободномолекулярный режим, затухание: decay≈{np.exp(-np.pi**2*T0*t_max**2):.3f})'
    )

    for ax, y_num, y_func, ylabel in zip(
        axs,
        [n_sol, u_sol, T_sol],
        [n_exact, u_exact, T_exact],
        ['n (density)', 'u (velocity)', 'T_code (temperature)'],
    ):
        ax.plot(x_fine, y_func(x_fine, t_max), 'k--', lw=1.5, label='exact (free mol.)')
        ax.scatter(x, y_num, s=10, color='tab:orange', zorder=3)
        ax.plot(x, y_num, color='tab:orange', lw=1, label=solver_name)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# ── Точка входа ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Сетка по ξ достаточно мелкая, чтобы хорошо разрешить Максвелл с T0=1
    # (±4σ при T0=1: σ=sqrt(T0/2)≈0.7, так что XI в [-4,4] покрывает всё)
    N_X_VALS  = [10, 20, 40, 80, 160]
    N_XI_VALS = [40] * len(N_X_VALS)   # ξ-сетка мельче, чем в гидро-тесте:
                                        # ошибка по ξ не должна маскировать
                                        # сходимость по x

    # ── RK2 ──
    print('\n=== RK2 — свободномолекулярный режим ===')
    err_rk2, t_rk2 = get_free_molecular_convergence(
        SolverRK, N_X_VALS, N_XI_VALS, n_ghost=2
    )
    print_convergence_table('SolverRK', N_X_VALS, err_rk2, t_rk2)

    # ── WENO5+RK3 ──
    print('\n=== WENO5+RK3 — свободномолекулярный режим ===')
    err_w5, t_w5 = get_free_molecular_convergence(
        WENO5RK3, N_X_VALS, N_XI_VALS, n_ghost=3
    )
    print_convergence_table('WENO5RK3', N_X_VALS, err_w5, t_w5)

    # ── Сравнительный L2-график ──
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title('L2-сходимость (свободномолекулярный режим, Kn=1e10)')
    h = 1.0 / np.array(N_X_VALS)
    ax.loglog(N_X_VALS, err_rk2[:, 2], 'b-o', label='RK2 (L2)')
    ax.loglog(N_X_VALS, err_w5[:, 2],  'r-s', label='WENO5+RK3 (L2)')
    ax.loglog(N_X_VALS, 0.5 * h**2,    'b--', lw=0.8, label='O(h²)')
    ax.loglog(N_X_VALS, 2.0 * h**5,    'r--', lw=0.8, label='O(h⁵)')
    ax.set_xlabel('n_x')
    ax.set_ylabel('L2 error')
    ax.legend()
    ax.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.show()

    # ── Детальный график для WENO5 на крупнейшей сетке ──
    plot_solution('WENO5RK3', n_x=80, n_xi=40, n_ghost=3)