from typing import Tuple

import numpy as np

from src.config.libloader import xp, cuda_is_available
from src.thermodynamics.model_properties import ModelProperties


class PropertyCalculator:

    @staticmethod
    def get_n(F, properties: ModelProperties):
        return xp.sum(F[properties.bc.n_ghost:-properties.bc.n_ghost], axis=(1, 2, 3)) * properties.dV

    @staticmethod
    def get_u1(F, n, properties: ModelProperties):
        return xp.sum(
            F[properties.bc.n_ghost:-properties.bc.n_ghost] * properties.xi[None, :, None, None],
            axis=(1, 2, 3)
        ) * properties.dV / xp.maximum(n, 1e-15)

    @staticmethod
    def get_T(F, n, u, properties: ModelProperties):

        F_slice = F[properties.bc.n_ghost:-properties.bc.n_ghost]
        E_x = xp.sum(
            F_slice * properties.xi_sq_x[None, :, :, :],
            axis=(1, 2, 3)
        )

        E_y = xp.sum(
            F_slice * properties.xi_sq_y[None, :, :, :],
            axis=(1, 2, 3)
        )

        E_z = xp.sum(
            F_slice * properties.xi_sq_z[None, :, :, :],
            axis=(1, 2, 3)
        )

        E = (E_x + E_y + E_z) * properties.dV

        return (2 / 3) * (E / xp.maximum(n, 1e-15) - u ** 2)

    @staticmethod
    def get_q(F, u, properties: ModelProperties):
        xi_shift = properties.xi_x[None, :, :, :] - u[:, None, None, None]

        c_sq = (
                xi_shift**2 +
                properties.xi_sq_y[None, :, :, :] +
                properties.xi_sq_z[None, :, :, :]
        )

        return 0.5 * xp.sum(
            xi_shift * c_sq * F[properties.bc.n_ghost:-properties.bc.n_ghost],
            axis=(1, 2, 3)
        ) * properties.dV


    @staticmethod
    def get_macros(F, properties: ModelProperties):

        F_slice = F[properties.bc.n_ghost:-properties.bc.n_ghost]

        n = xp.sum(F_slice, axis=(1, 2, 3)) * properties.dV

        u_num = xp.sum(
            F_slice * properties.xi_x[None, :, :, :],
            axis=(1, 2, 3)
        ) * properties.dV

        u = u_num / xp.maximum(n, 1e-15)

        E_x = xp.sum(
            F_slice * properties.xi_sq_x[None, :, :, :],
            axis=(1, 2, 3)
        )

        E_y = xp.sum(
            F_slice * properties.xi_sq_y[None, :, :, :],
            axis=(1, 2, 3)
        )

        E_z = xp.sum(
            F_slice * properties.xi_sq_z[None, :, :, :],
            axis=(1, 2, 3)
        )

        E = (E_x + E_y + E_z) * properties.dV

        T = (2.0 / 3.0) * (E / xp.maximum(n, 1e-15) - u ** 2)

        xi_shift = properties.xi_x[None, :, :, :] - u[:, None, None, None]

        c_sq = (
                xi_shift ** 2 +
                properties.xi_sq_y[None, :, :, :] +
                properties.xi_sq_z[None, :, :, :]
        )

        q = 0.5 * xp.sum(
            xi_shift * c_sq * F_slice,
            axis=(1, 2, 3)
        ) * properties.dV

        return n, u, T, q

    @staticmethod
    def get_solution_macros(F, properties: ModelProperties) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n, u, T, q = PropertyCalculator.get_macros(F, properties)
        if cuda_is_available:
            return xp.asnumpy(n), xp.asnumpy(u), xp.asnumpy(T), xp.asnumpy(q)
        else:
            return n, u, T, q

    @staticmethod
    def get_mu(T, properties: ModelProperties):
        return T**properties.w

    @staticmethod
    def get_nu(n, T, properties: ModelProperties):
        return n * T ** (1 - properties.w) * 0.9 / properties.Kn

    @staticmethod
    def get_fS(F, properties: ModelProperties):

        n, u, T, q = PropertyCalculator.get_macros(F, properties)

        n4 = n[:, None, None, None]
        u4 = u[:, None, None, None]
        T4 = xp.maximum(T[:, None, None, None], 1e-12)
        q4 = q[:, None, None, None]

        sqrtT = xp.sqrt(T4)

        tx = (properties.xi_x[None, :, :, :] - u4) / sqrtT
        ty = properties.xi_y[None, :, :, :] / sqrtT
        tz = properties.xi_z[None, :, :, :] / sqrtT

        c_sq = tx * tx + ty * ty + tz * tz

        M = xp.exp(-c_sq)

        Z = xp.sum(M, axis=(1, 2, 3), keepdims=True) * properties.dV

        fM = n4 * M / Z

        S = 2 * q4 / (xp.maximum(n4, 1e-12) * T4 ** 1.5)

        shakhov = (4 / 5) * (1 - properties.Pr) * S * tx * (c_sq - 2.5)

        return fM * (1 + shakhov)