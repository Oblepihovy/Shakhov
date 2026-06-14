from src.advection_solvers.base import *
from src.config.libloader import xp


class SolverGodunov(Solver):
    __name__ = "Godunov"

    def _step(self, F, t, tau, properties: ModelProperties, prop_calc):
        properties.bc.apply(F, t, properties, prop_calc)

        F_l = F[:-1]
        F_r = F[1:]

        xi = properties.xi[None, :, None, None]
        Ws = W_god(F_l, F_r, xi)

        # ИСПРАВЛЕНИЕ: dx[1:-1] чтобы совпадало по размеру с F[1:-1].
        # Ранее dx[:] имел размер N_cells, а F[1:-1] — N_cells-2, что вызывало
        # broadcast-ошибку. «Обходной» способ +1 в F.shape был именно из-за этого.
        dx = properties.mesh.get_dx()[1:-1, None, None, None]

        F[1:-1] += (tau / dx) * xi * (Ws[:-1] - Ws[1:])

    def calculate_layer(self, F, t, tau, properties: ModelProperties, prop_calc):
        self._step(F, t, tau, properties, prop_calc)
        super()._calculate_collisions(F, tau, properties, prop_calc)