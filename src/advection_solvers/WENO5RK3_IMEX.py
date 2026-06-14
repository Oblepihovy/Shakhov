from src.advection_solvers.base import Solver
from src.config.libloader import xp
from src.thermodynamics.model_properties import ModelProperties


class WENO5RK3(Solver):
    __name__ = "WENO5RK3"

    def __init__(self, eps=1e-12):
        self.eps = eps
        self._buffers_allocated = False

    # ------------------------------------------------------------------ WENO5
    def _weno5_left(self, f):
        eps = self.eps
        im2, im1, i0, ip1, ip2 = f[:-5], f[1:-4], f[2:-3], f[3:-2], f[4:-1]

        beta0 = (13/12)*(im2 - 2*im1 + i0)**2  + 0.25*(im2 - 4*im1 + 3*i0)**2
        beta1 = (13/12)*(im1 - 2*i0  + ip1)**2 + 0.25*(im1 - ip1)**2
        beta2 = (13/12)*(i0  - 2*ip1 + ip2)**2 + 0.25*(3*i0 - 4*ip1 + ip2)**2

        d0, d1, d2 = 0.1, 0.6, 0.3
        a0 = d0 / (eps + beta0)**2
        a1 = d1 / (eps + beta1)**2
        a2 = d2 / (eps + beta2)**2
        asum = a0 + a1 + a2
        w0, w1, w2 = a0/asum, a1/asum, a2/asum

        q0 = ( 1/3)*im2 - (7/6)*im1 + (11/6)*i0
        q1 = -(1/6)*im1 + (5/6)*i0  + ( 1/3)*ip1
        q2 = ( 1/3)*i0  + (5/6)*ip1 - ( 1/6)*ip2

        return w0*q0 + w1*q1 + w2*q2

    def _weno5_right(self, f):
        eps = self.eps
        ip2, ip1, i0, im1, im2 = f[5:], f[4:-1], f[3:-2], f[2:-3], f[1:-4]

        beta0 = (13/12)*(ip2 - 2*ip1 + i0)**2  + 0.25*(ip2 - 4*ip1 + 3*i0)**2
        beta1 = (13/12)*(ip1 - 2*i0  + im1)**2 + 0.25*(ip1 - im1)**2
        beta2 = (13/12)*(i0  - 2*im1 + im2)**2 + 0.25*(3*i0 - 4*im1 + im2)**2

        d0, d1, d2 = 0.3, 0.6, 0.1
        a0 = d0 / (eps + beta0)**2
        a1 = d1 / (eps + beta1)**2
        a2 = d2 / (eps + beta2)**2
        asum = a0 + a1 + a2
        w0, w1, w2 = a0/asum, a1/asum, a2/asum

        q0 = ( 1/3)*ip2 - (7/6)*ip1 + (11/6)*i0
        q1 = -(1/6)*ip1 + (5/6)*i0  + ( 1/3)*im1
        q2 = ( 1/3)*i0  + (5/6)*im1 - ( 1/6)*im2

        return w0*q0 + w1*q1 + w2*q2

    # ---------------------------------------------------------------- буферы
    def _alloc_buffers(self, F):
        self._rhs = xp.zeros_like(F)
        self._F0  = xp.zeros_like(F)
        self._F1  = xp.zeros_like(F)   # ← pre-allocated, раньше создавался каждый шаг
        self._F2  = xp.zeros_like(F)   # ← аналогично
        self._buffers_allocated = True

    # -------------------------------------------------------------- один шаг
    def _step(self, F, t, tau, properties, prop_calc):
        if not self._buffers_allocated:
            self._alloc_buffers(F)

        properties.bc.apply(F, t, properties, prop_calc)

        xi    = properties.xi[None, :, None, None]
        alpha = xp.abs(xi)

        f_p = 0.5 * (xi + alpha) * F
        f_m = 0.5 * (xi - alpha) * F

        flux = self._weno5_left(f_p) + self._weno5_right(f_m)

        dx = properties.mesh.get_dx()[:, None, None, None]

        rhs = self._rhs
        rhs.fill(0)

        flux_diff = flux[1:] - flux[:-1]

        # ИСПРАВЛЕНИЕ: start = n_ghost, а не захардкоженная 3.
        # WENO5 требует n_ghost ghost-ячеек на каждой стороне (стенсил ±2).
        # При n_ghost=3 start=3 ровно совпадает с первой физической ячейкой,
        # и ВСЕ физические ячейки получают правильный advection-update.
        # При n_ghost=2 (как было раньше) первая и последняя физ. ячейки
        # исключались из обновления.
        start = properties.bc.n_ghost
        end   = start + flux_diff.shape[0]

        rhs[start:end] = -flux_diff / dx[start:end]

        return rhs

    # ----------------------------------------------------------- SSP-RK3 шаг
    def calculate_layer(self, F, t, tau, properties: ModelProperties, prop_calc):
        """
        IMEX-SSP-RK3 для уравнения Больцмана–Шахова:
          ∂F/∂t = L_adv(F) + L_coll(F)

        Наивный вариант (L_adv → BGK в конце) — расщепление Ли–Троттера:
        на стадиях 2–3 SSP-RK3 F₁, F₂ являются НЕРАВНОВЕСНЫМИ.
        Поток импульса ∫ξx²F₁dξ содержит поправку O(τ) от 3-го момента,
        что даёт суммарную погрешность O(τ) = O(h) по импульсу → rate=1.

        Фикс (этот вариант): применяем BGK-коллизию к промежуточным стадиям,
        возвращая F₁ и F₂ на максвеллианово многообразие перед вычислением k2, k3.
        В жёстком пределе (Kn·ν·τ >> 1) коллизия ≡ проекции на максвеллиан,
        поэтому стадии 2 и 3 снова используют правильные эйлеровские потоки.
        Итоговая схема эквивалентна SSP-RK3 на уравнениях Эйлера → rate≈3.

        Стоимость: 3 вызова BGK вместо 1. Для жёсткого BGK каждый вызов —
        это get_macros + init_F_vectorized, что сопоставимо с одним шагом адвекции.
        """
        if not self._buffers_allocated:
            self._alloc_buffers(F)

        F0 = self._F0
        F1 = self._F1
        F2 = self._F2

        F0[:] = F

        # Стадия 1: адвекция из максвеллиана F0
        k1    = self._step(F, t, tau, properties, prop_calc)
        F1[:] = F0 + tau * k1
        # Проецируем F1 на максвеллиан (BGK для стадии 1)
        super()._calculate_collisions(F1, tau, properties, prop_calc)

        # Стадия 2: адвекция из максвеллиана F1
        k2    = self._step(F1, t + tau, tau, properties, prop_calc)
        F2[:] = 0.75 * F0 + 0.25 * (F1 + tau * k2)
        # Проецируем F2 на максвеллиан (BGK для стадии 2)
        super()._calculate_collisions(F2, tau, properties, prop_calc)

        # Стадия 3: адвекция из максвеллиана F2
        k3    = self._step(F2, t + 2*tau, tau, properties, prop_calc)
        F[:]  = (1/3) * F0 + (2/3) * (F2 + tau * k3)

        # Финальная коллизия
        super()._calculate_collisions(F, tau, properties, prop_calc)