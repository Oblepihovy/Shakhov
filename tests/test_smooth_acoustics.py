"""
test_conv_hydro_smooth.py
--------------------------
Тест сходимости на гладком решении — гидродинамический режим (Kn = 1e-5).

Физика:
  При Kn → 0 S-модель (Shakhov) вырождается в уравнения Эйлера.
  Для малого возмущения EPS ≪ 1 линеаризованные уравнения Эйлера дают
  правую бегущую акустическую волну:

      n(x, t) = N0 + EPS · sin(2π·(x − c_s·t))
      u(x, t) = U0 + EPS · (c_s/N0) · sin(2π·(x − c_s·t))
      T(x, t) = T0 + EPS · ((γ−1)·T0/N0) · sin(2π·(x − c_s·t))

  Скорость звука (кинетические единицы, распределение ~ exp(−ξ²/T)):
      c_s = sqrt(γ·T0/2),   γ = 5/3  →  c_s = sqrt(5·T0/6)

Стратегия «возврат к начальному условию»:
  Дискретная квадратура по ξ даёт c_s^h ≠ c_s (аналитическая).
  Если сравнивать с n_exact(x, t_end) при t_end ≠ k·T_period, фазовый
  сдвиг (c_s − c_s^h)·t_end создаёт константный «пол» ошибки, который
  не убывает при измельчении сетки по x — именно это и наблюдается
  при WENO5 на мелких сетках.

  Решение: t_end = целое число периодов T_period = L/c_s^h.
  Тогда n(x, t_end) = n(x, 0) = начальное условие — точный эталон
  без какой-либо зависимости от c_s^h.

  c_s^h измеряется один раз на мелкой сетке (n_x=640, n_xi=N_XI_REF)
  по максимуму функции n(x,t) методом «первого пересечения» через
  трассировку фазы. На практике используется аналитическая формула
  с поправкой: берётся БЛИЖАЙШЕЕ t_end, кратное T_period.

Почему работает оператор столкновений:
  - Kn=1e-5: τ = Kn·μ/p ≪ t_max → f экспоненциально близко к
    максвелловскому распределению → макро-поля подчиняются Эйлеру.
  - EPS=1e-3: нелинейный пол O(ε²) ~ 1e-6 ≪ ошибки сетки N_x=10..640.

Ожидаемые порядки по L2:
  Godunov                ~ 1
  Kolgan                 ~ 2
  RK2                    ~ 2
  WENO5 + SSP-RK3        ~ 3..5  (ограничен SSP-RK3 при τ~h)

Параметры: N_ξ = 40 (фиксирован), CFL = 0.45 (Kolgan/RK2), 0.8 (остальные)
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


# ── Параметры ──────────────────────────────────────────────────────────────────

EPS    = 1e-4   # амплитуда (линейный режим: нелинейный пол O(ε²) ~ 1e-6)
N0     = 1.0
U0     = 0.0
T0     = 1.0    # T_code
TD_KN  = 1e-5   # гидродинамический режим: столкновения доминируют

GAMMA  = 5.0 / 3.0
C_S    = float(np.sqrt(GAMMA * T0 / 2.0))   # аналитическая скорость звука

L_DOMAIN = float(abs(X_RIGHT - X_LEFT))     # длина области (обычно 1.0)

# Период: время, за которое волна обходит область ровно N_PERIODS раз.
# При t = N_PERIODS * T_period решение совпадает с начальным условием точно,
# независимо от дискретной c_s^h — это и есть наш «нулевой» эталон.
N_PERIODS = 1
T_PERIOD  = L_DOMAIN / C_S      # аналитический период (приближённый)

# Окончательное t_end подбираем так, чтобы c_s^h * t_end ≈ N_PERIODS * L.
# Поправку вносим в get_hydro_smooth_convergence через аргумент t_max.
# Здесь просто сохраняем T_PERIOD для справки; реальный t_end считается ниже.


# ── Начальные условия ──────────────────────────────────────────────────────────

def _F_BEG_N(x):
    """Плотность: синусоидальное возмущение."""
    return N0 + EPS * xp.sin(2.0 * xp.pi * x)

def _F_BEG_U(x):
    """Скорость: акустическая связь δu = c_s/N0 · δn."""
    return xp.zeros_like(x) + U0 + EPS * (C_S / N0) * xp.sin(2.0 * xp.pi * x)

def _F_BEG_T(x):
    """Температура: адиабатическая связь δT = (γ−1)·T0/N0 · δn."""
    return xp.zeros_like(x) + T0 + EPS * ((GAMMA - 1.0) * T0 / N0) * xp.sin(
        2.0 * xp.pi * x
    )


# ── Точное решение ─────────────────────────────────────────────────────────────

def n_ic(x):
    """Начальное условие по плотности — оно же эталон при t = k·T_period."""
    return N0 + EPS * np.sin(2.0 * np.pi * x)

def n_exact(x, t):
    """
    Аналитическая бегущая волна (используется только для графиков профиля).
    При t = k·T_period совпадает с n_ic(x).
    """
    return N0 + EPS * np.sin(2.0 * np.pi * (x - C_S * t))


# ── Измерение дискретной скорости звука c_s^h ──────────────────────────────────

def measure_cs_h(advection_solver_cls, n_xi, n_x_ref=160, n_ghost=3,
                 CFL=0.8, t_probe=None):
    """
    Запускает расчёт на n_x_ref × n_xi до t_probe ≈ T_PERIOD/4,
    находит положение максимума n(x,t) и вычисляет c_s^h = x_max / t_probe.

    Возвращает c_s^h (float).
    """
    if t_probe is None:
        t_probe = T_PERIOD / 4.0   # четверть аналитического периода

    dtype = xp.float64
    model_config = {
        'X_LEFT':  X_LEFT,  'X_RIGHT':  X_RIGHT,
        'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
        'F_BEG_N': _F_BEG_N, 'F_BEG_U': _F_BEG_U, 'F_BEG_T': _F_BEG_T,
        'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W, 'dtype': dtype,
    }
    bc   = PeriodicBoundaryCondition(n_ghost)
    mesh = UnadaptableMesh(
        xp.linspace(X_LEFT, X_RIGHT, n_x_ref + 1, endpoint=True),
        bc.n_ghost,
    )
    prop = ModelProperties(model_config, mesh, bc)
    st   = ModelState(prop, model_config)
    ShakhovSolver(st, prop, advection_solver_cls()).calculate(CFL, t_probe)

    x_centers = prop.mesh.get_centers()[bc.n_ghost:-bc.n_ghost]
    if cuda_is_available:
        x_centers = xp.asnumpy(x_centers)
    n_sol, _, _, _ = PropertyCalculator.get_solution_macros(st.F, prop)

    # Максимум n(x,t_probe): при t=T/4 он сдвинут на c_s*T/4 = L/4 от x=0.
    # x_max_0 = 0.25 (четверть периода от начала синуса, который стартует в x=0.75)
    # Используем argmax для надёжности.
    idx_max  = int(np.argmax(n_sol))
    x_max    = float(x_centers[idx_max])

    # Начальный максимум синуса: sin(2πx) максимален при x=0.25
    x_max_ic = 0.25
    shift    = (x_max - x_max_ic) % L_DOMAIN  # сдвиг волны за t_probe
    cs_h     = shift / t_probe
    return cs_h


# ── Основная функция ───────────────────────────────────────────────────────────

def get_hydro_smooth_convergence(
    advection_solver_cls,
    n_x_vals,
    n_xi_vals,
    n_ghost=3,
    CFL=0.8,
    n_periods=N_PERIODS,
    cs_h=None,
):
    """
    Параметры
    ---------
    advection_solver_cls : класс солвера
    n_x_vals             : список числа физических ячеек по x
    n_xi_vals            : список числа ячеек по ξ
    n_ghost              : число ghost-ячеек
    CFL                  : число Куранта
    n_periods            : число периодов (t_end = n_periods * T_period_h)
    cs_h                 : дискретная скорость звука; если None — измеряется
                           автоматически на n_x=160 с первым n_xi из n_xi_vals

    Возвращает
    ----------
    errors : ndarray (len(n_x_vals), 3) — [L_inf, L1, L2] по плотности n
    times  : ndarray (len(n_x_vals),)   — времена расчёта в секундах
    t_end  : float — использованное конечное время
    """
    if len(n_x_vals) != len(n_xi_vals):
        raise ValueError('n_x_vals и n_xi_vals должны иметь одинаковую длину')

    # Измеряем c_s^h один раз — для первого n_xi в списке
    if cs_h is None:
        print(f'  Измеряем c_s^h (n_xi={n_xi_vals[0]}, n_x=160)...', end=' ', flush=True)
        cs_h = measure_cs_h(advection_solver_cls, n_xi_vals[0],
                            n_x_ref=160, n_ghost=n_ghost, CFL=CFL)
        print(f'c_s^h = {cs_h:.6f}  (аналит. {C_S:.6f},  δ={abs(cs_h-C_S):.2e})')

    t_end = n_periods * L_DOMAIN / cs_h
    print(f'  t_end = {n_periods} * T_period_h = {t_end:.6f}')

    dtype = xp.float64
    calculation_times  = np.zeros(len(n_x_vals))
    calculation_errors = np.zeros((len(n_x_vals), 3))

    # Эталон — начальное условие (точное при t = k·T_period_h)
    # Вычисляем заранее на мелком равномерном узле; для каждой сетки
    # пересчитывается по x_centers конкретной сетки.

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
            bc.n_ghost,
        )

        adv_solver = advection_solver_cls()
        properties = ModelProperties(model_config, mesh, bc)
        state      = ModelState(properties, model_config)
        solver     = ShakhovSolver(state, properties, adv_solver)

        t1 = time.time()
        solver.calculate(CFL, t_end)
        t2 = time.time()
        calculation_times[i] = t2 - t1

        x_centers = properties.mesh.get_centers()[bc.n_ghost:-bc.n_ghost]
        if cuda_is_available:
            x_centers = xp.asnumpy(x_centers)

        n_sol, _, _, _ = PropertyCalculator.get_solution_macros(state.F, properties)

        # Эталон: начальное условие на тех же узлах сетки
        n_ref = n_ic(x_centers)

        l_inf = float(L_sup(x_centers, n_ref, n_sol))
        l1    = float(L1(x_centers, n_ref, n_sol))
        l2    = float(L2(x_centers, n_ref, n_sol))

        print(
            f'  n_x={n_x:4d}  n_xi={n_xi:3d}  t={t2 - t1:6.2f}s'
            f'  L_inf={l_inf:.3e}  L1={l1:.3e}  L2={l2:.3e}'
        )
        calculation_errors[i] = [l_inf, l1, l2]

    return calculation_errors, calculation_times, t_end



# ── Таблица сходимости ─────────────────────────────────────────────────────────

def print_convergence_table(solver_name, n_x_vals, errors, times, t_end=None):
    w = 78
    t_label = f't={t_end:.4f}' if t_end is not None else ''
    print()
    print('=' * w)
    print(f'  {solver_name}  (гидродинамический режим, Kn=1e-5, {t_label})')
    print('-' * w)
    print(
        f'  {"n_x":>5}  {"L_inf":>10}  {"rate":>5}  '
        f'{"L1":>10}  {"rate":>5}  {"L2":>10}  {"rate":>5}  {"t,s":>6}'
    )
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
            ratio     = n_x_vals[i] / n_x_vals[i - 1]
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


# ── График профиля ─────────────────────────────────────────────────────────────

def plot_profile(solver_name_cls_ghost, n_x, n_xi, t_end=None, CFL=0.8):
    """
    Детальный график n(x): численное vs точное.
    solver_name_cls_ghost — тройка (имя, класс, n_ghost).
    t_end: если None — измеряется автоматически.
    """
    dtype = xp.float64
    name, cls, n_ghost = solver_name_cls_ghost

    if t_end is None:
        cs_h  = measure_cs_h(cls, n_xi, n_ghost=n_ghost, CFL=CFL)
        t_end = N_PERIODS * L_DOMAIN / cs_h

    model_config = {
        'X_LEFT':  X_LEFT,  'X_RIGHT':  X_RIGHT,
        'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
        'F_BEG_N': _F_BEG_N, 'F_BEG_U': _F_BEG_U, 'F_BEG_T': _F_BEG_T,
        'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W, 'dtype': dtype,
    }
    bc   = PeriodicBoundaryCondition(n_ghost)
    mesh = UnadaptableMesh(
        xp.linspace(X_LEFT, X_RIGHT, n_x + 1),
        bc.n_ghost,
    )
    prop = ModelProperties(model_config, mesh, bc)
    st   = ModelState(prop, model_config)
    ShakhovSolver(st, prop, cls()).calculate(CFL, t_end)

    x = prop.mesh.get_centers()[bc.n_ghost:-bc.n_ghost]
    if cuda_is_available:
        x = xp.asnumpy(x)
    n_sol, _, _, _ = PropertyCalculator.get_solution_macros(st.F, prop)

    x_fine = np.linspace(X_LEFT, X_RIGHT, 2000)

    FSIZE = 16
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_fine, n_ic(x_fine), 'k-', lw=1.5,
            label='Эталон (нач. условие = решение при t=k·T)')
    ax.plot(x, n_sol, 'o-', ms=3, lw=1, color='tab:orange', label=name)
    ax.set_xlabel('x', fontsize=FSIZE)
    ax.set_ylabel('n(x)', fontsize=FSIZE)
    ax.set_title(
        f'{name},  $N_x={n_x}$,  $N_\\xi={n_xi}$,  $t={t_end:.4f}$,  Kn=10⁻⁵',
        fontsize=FSIZE,
    )
    ax.tick_params(axis='both', labelsize=FSIZE - 2)
    ax.legend(fontsize=FSIZE - 2)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ── Точка входа ────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    N_X_VALS  = [10, 20, 40, 80, 160, 320, 640]
    N_XI_VALS = [20] * len(N_X_VALS)

    print(f'\nc_s (аналит.) = {C_S:.6f}')
    print(f'T_period (аналит.) = {T_PERIOD:.6f}')
    print(f'Нелинейный пол O(ε²) ~ {EPS**2:.1e}')
    print(
        '\nСтратегия: t_end = N_PERIODS * L / c_s^h  '
        '→ эталон = начальное условие (независимо от c_s^h)'
    )

    # ── Godunov ───────────────────────────────────────────────────────────────
    print('\n=== Godunov ===')
    err_god, t_god, t_end_god = get_hydro_smooth_convergence(
        SolverGodunov, N_X_VALS, N_XI_VALS, n_ghost=2, CFL=0.8
    )
    print_convergence_table('SolverGodunov', N_X_VALS, err_god, t_god, t_end_god)

    # ── Kolgan ────────────────────────────────────────────────────────────────
    print('\n=== Kolgan ===')
    err_klg, t_klg, t_end_klg = get_hydro_smooth_convergence(
        SolverKolgan, N_X_VALS, N_XI_VALS, n_ghost=2, CFL=0.45
    )
    print_convergence_table('SolverKolgan', N_X_VALS, err_klg, t_klg, t_end_klg)

    # ── RK2 ───────────────────────────────────────────────────────────────────
    print('\n=== RK2 ===')
    err_rk2, t_rk2, t_end_rk2 = get_hydro_smooth_convergence(
        SolverRK, N_X_VALS, N_XI_VALS, n_ghost=3, CFL=0.45
    )
    print_convergence_table('SolverRK', N_X_VALS, err_rk2, t_rk2, t_end_rk2)

    # ── WENO5+RK3 ─────────────────────────────────────────────────────────────
    print('\n=== WENO5+RK3 ===')
    err_w5, t_w5, t_end_w5 = get_hydro_smooth_convergence(
        WENO5RK3, N_X_VALS, N_XI_VALS, n_ghost=3, CFL=0.8
    )
    print_convergence_table('WENO5RK3', N_X_VALS, err_w5, t_w5, t_end_w5)

    # ── L2-график ─────────────────────────────────────────────────────────────
    FSIZE = 16
    h = 1.0 / np.array(N_X_VALS)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(
        'L2-сходимость, гидродинамический режим, Kn=1e-5',
        fontsize=FSIZE,
    )
    ax.loglog(N_X_VALS, err_god[:, 2], 'g-^', label='Godunov')
    ax.loglog(N_X_VALS, err_klg[:, 2], 'm-v', label='Kolgan')
    ax.loglog(N_X_VALS, err_rk2[:, 2], 'b-o', label='RK2')
    ax.loglog(N_X_VALS, err_w5[:,  2], 'r-s', label='WENO5+RK3')

    # эталонные наклоны
    ax.loglog(N_X_VALS, err_god[0, 2] * (h / h[0])**1, 'g--', lw=0.9, label='O(h¹)')
    ax.loglog(N_X_VALS, err_rk2[0, 2] * (h / h[0])**2, 'b--', lw=0.9, label='O(h²)')
    ax.loglog(N_X_VALS, err_w5[0,  2] * (h / h[0])**5, 'r--', lw=0.9, label='O(h⁵)')
    ax.loglog(N_X_VALS, err_w5[1,  2] * (h / h[1])**3, 'r:',  lw=0.9, label='O(h³)')

    ax.set_xlabel('$N_x$', fontsize=FSIZE)
    ax.set_ylabel('$\\varepsilon_{L_2}$', fontsize=FSIZE)
    ax.tick_params(axis='both', labelsize=FSIZE - 2)
    ax.legend(fontsize=FSIZE - 2, ncol=2)
    ax.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.show()

    # ── Профиль ───────────────────────────────────────────────────────────────
    # plot_profile(('WENO5+RK3', WENO5RK3, 3), n_x=160, n_xi=40, t_end=t_end_w5)
    # plot_profile(('Kolgan', SolverKolgan, 2), n_x=160, n_xi=40, t_end=t_end_klg)