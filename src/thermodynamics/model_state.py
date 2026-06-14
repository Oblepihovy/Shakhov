from src.thermodynamics.model_properties import ModelProperties
from src.config.libloader import xp


class ModelState:

    def __init__(self, properties: ModelProperties, config: dict):
        self.F = None
        self.init_conditions(properties, config)

    def init_conditions(self, properties: ModelProperties, config: dict):
        ng = properties.bc.n_ghost

        # get_centers() возвращает центры ВСЕХ ячеек (включая ghost),
        # длина = len(mesh.x) - 1  (число ячеек = число промежутков между узлами)
        x_centers = properties.mesh.get_centers()
        x_phys    = x_centers[ng:-ng]       # только физические ячейки

        n_cells = len(x_centers)            # правильный размер F по оси x
        n_xi    = len(properties.xi)

        n = xp.zeros(n_cells, dtype=config['dtype'])
        u = xp.zeros(n_cells, dtype=config['dtype'])
        T = xp.zeros(n_cells, dtype=config['dtype'])

        n[ng:-ng] = config['F_BEG_N'](x_phys)
        u[ng:-ng] = config['F_BEG_U'](x_phys)
        T[ng:-ng] = config['F_BEG_T'](x_phys)

        self.F = xp.zeros((n_cells, n_xi, n_xi, n_xi), dtype=config['dtype'])
        self.F[ng:-ng] = self.init_F_vectorized(n, u, T, properties, ng)

    @staticmethod
    def init_F_vectorized(n, u, T, properties: ModelProperties, n_ghost):
        n_loc = n[n_ghost:-n_ghost]
        u_loc = u[n_ghost:-n_ghost]
        T_loc = T[n_ghost:-n_ghost]

        xi  = properties.xi
        dxi = properties.xi_cell_size

        M1 = xp.exp(-(xi[None, :] - u_loc[:, None]) ** 2 / T_loc[:, None])
        M2 = xp.exp(-(xi[None, :] ** 2)              / T_loc[:, None])
        M3 = xp.exp(-(xi[None, :] ** 2)              / T_loc[:, None])

        Z1 = xp.sum(M1, axis=1) * dxi
        Z2 = xp.sum(M2, axis=1) * dxi
        Z3 = xp.sum(M3, axis=1) * dxi
        Z  = Z1 * Z2 * Z3

        F = (
            n_loc[:, None, None, None]
            * M1[:, :, None, None]
            * M2[:, None, :, None]
            * M3[:, None, None, :]
            / Z[:, None, None, None]
        )
        return F

    def get_F(self):
        return self.F

    def set_F(self, F):
        if F.shape != self.F.shape:
            raise ValueError(
                f"Shape mismatch: F.shape={F.shape} != self.F.shape={self.F.shape}"
            )
        self.F = F