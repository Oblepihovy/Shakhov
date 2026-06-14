from abc import ABC, abstractmethod
from src.config.libloader import xp


class Mesh(ABC):
    __name__ = "Mesh"

    def __init__(self, x, n_ghost):
        dx_l, dx_r = x[1] - x[0], x[-1] - x[-2]
        idx = xp.arange(1, n_ghost + 1)

        l_ghost = x[0] - dx_l*idx[::-1]
        r_ghost = x[-1] + dx_r * idx

        self.x = xp.concatenate([l_ghost, x, r_ghost])
        self.dx = self.x[1:] - self.x[:-1]

    def get_centers(self):
        return self.x[:-1] + self.get_dx()/2

    def get_dx(self):
        return self.dx

    def get_name(self):
        return self.__name__

    @abstractmethod
    def update(self, F, prop_calc, props):
        pass


class UnadaptableMesh(Mesh):
    __name__ = "Unadaptable Mesh"
    def __init__(self, x, n_ghost):
        super().__init__(x, n_ghost)

    def update(self, F, prop_calc, props):
        pass


class RezoningMesh(Mesh):
    __name__ = "Rezoning Mesh"

    def __init__(self, x, n_ghost, alpha=0.5):
        """
        :param f_func: lambda dn, du, dT, dq: monitor function
        """
        eps = 1e-6
        self.f_func = lambda dn, n: (
            1 + 5 * xp.tanh(xp.abs(dn) / (xp.abs(n) + eps))
        )
        self.alpha = alpha
        super().__init__(x, n_ghost)

    def get_dx(self):
        return self.dx

    def update(self, F, prop_calc, props):
        ng = props.bc.n_ghost
        n, u, T, q = prop_calc.get_macros(F, props)

        dx_phys = self.dx[ng:-ng]
        print('До адаптации:', xp.sum(n * dx_phys))

        dn = xp.abs(xp.diff(n)) / dx_phys[:-1]
        du = xp.abs(xp.diff(u)) / dx_phys[:-1]
        dT = xp.abs(xp.diff(T)) / dx_phys[:-1]
        dq = xp.abs(xp.diff(q)) / dx_phys[:-1]

        # выравниваем размер до числа ячеек
        dn = xp.pad(dn, (1, 0), mode='edge')
        du = xp.pad(du, (1, 0), mode='edge')
        dT = xp.pad(dT, (1, 0), mode='edge')
        dq = xp.pad(dq, (1, 0), mode='edge')

        M = self.f_func(dn, n)

        if not xp.max(M) / xp.mean(M) > 1.2:
            return

        print('Adapting the mesh...')
        for _ in range(6):
            M[1:-1] = 0.25 * M[:-2] + 0.5 * M[1:-1] + 0.25 * M[2:]

        S = xp.cumsum(M * dx_phys)
        S = S / S[-1]

        N_phys = len(S)
        S_uniform = xp.linspace(0.0, 1.0, N_phys)

        x_centers_old = self.get_centers()[ng:-ng]
        x_centers_new = xp.interp(S_uniform, S, x_centers_old)
        x_new_phys = xp.zeros(N_phys + 1, dtype=self.x.dtype)
        x_new_phys[1:-1] = 0.5 * (x_centers_new[:-1] + x_centers_new[1:])
        x_new_phys[0] = x_centers_new[0] - (x_new_phys[1] - x_centers_new[0])
        x_new_phys[-1] = x_centers_new[-1] + (x_centers_new[-1] - x_new_phys[-2])

        dx_l = x_new_phys[1] - x_new_phys[0]
        dx_r = x_new_phys[-1] - x_new_phys[-2]

        idx = xp.arange(1, ng + 1)
        l_ghost = x_new_phys[0] - dx_l * idx[::-1]
        r_ghost = x_new_phys[-1] + dx_r * idx

        x_full_new = xp.concatenate([l_ghost, x_new_phys, r_ghost])
        x_full_blended = (1 - self.alpha) * self.x + self.alpha * x_full_new

        x_centers_full_old = self.get_centers()
        x_centers_full_blended = x_full_blended[:-1] + (x_full_blended[1:] - x_full_blended[:-1]) / 2

        idxs = xp.searchsorted(x_centers_full_old, x_centers_full_blended) - 1
        idxs = xp.clip(idxs, 0, len(x_centers_full_old) - 2)

        x0 = x_centers_full_old[idxs]
        x1 = x_centers_full_old[idxs + 1]

        w = (x_centers_full_blended - x0) / (x1 - x0 + 1e-14)
        w = w[:, None, None, None]

        F0 = F[idxs]
        F1 = F[idxs + 1]

        F_new = (1 - w) * F0 + w * F1

        self.x = x_full_blended
        self.dx = self.x[1:] - self.x[:-1]
        F[:] = F_new

        n, u, T, q = prop_calc.get_macros(F, props)
        print('После адаптации:', xp.sum(n * dx_phys))


def graded_linspace(xp, n_points, a=0.01, length=1.0):
    n_cells = n_points - 1

    d = (2 * length / n_cells - 2 * a) / (n_cells - 1)

    k = xp.arange(n_cells)
    dx = a + k * d

    x = xp.zeros(n_points)
    x[1:] = xp.cumsum(dx)

    return x
