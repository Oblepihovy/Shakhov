import numpy as np

def euler_exact(
        x: np.ndarray, x0: float,
        n_l: float, n_r: float,
        u_l: float, u_r: float,
        T_l: float, T_r: float,
        t: float,
        gamma: float = 5.0 / 3.0,
        tol: float = 1e-6,
        max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact solution of the Sod (Riemann) shock tube problem.

    Follows the algorithm of Toro, "Riemann Solvers and Numerical Methods
    for Fluid Dynamics", 3rd ed., Chapter 4.

    p = n * T  (ideal gas, k_B = m = 1).

    Parameters
    ----------
    x        : 1-D grid of positions.
    x0       : initial discontinuity location.
    n_l/r    : number density  left / right.
    u_l/r    : bulk velocity   left / right.
    T_l/r    : temperature     left / right.
    t        : time (t > 0).
    gamma    : adiabatic index (default 5/3).
    tol      : Newton convergence tolerance on relative pressure change.
    max_iter : maximum Newton iterations.

    Returns
    -------
    n, u, T  : np.ndarray of shape (len(x),).
    """

    # ------------------------------------------------------------------ #
    #  Primitive variables                                                 #
    # ------------------------------------------------------------------ #
    p_l = n_l * T_l
    p_r = n_r * T_r
    c_l = np.sqrt(gamma * T_l)
    c_r = np.sqrt(gamma * T_r)

    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    g4 = 2.0 / gm1  # 2/(gamma-1)
    g5 = 2.0 / gp1  # 2/(gamma+1)
    g6 = gm1 / gp1  # (gamma-1)/(gamma+1)
    g7 = gm1 / (2.0 * gamma)  # (gamma-1)/(2*gamma)
    g3 = 2.0 * gamma / gm1  # 2*gamma/(gamma-1)

    # ------------------------------------------------------------------ #
    #  Toro eq. (4.6): pressure function f(p, p_k, rho_k, c_k)           #
    #  and its derivative f'(p)  — eq. (4.37)                            #
    # ------------------------------------------------------------------ #
    def _f(p, p_k, rho_k, c_k):
        if p > p_k:  # shock (Hugoniot)
            A = g5 / rho_k
            B = g6 * p_k
            sqrt_AB = np.sqrt(A / (p + B))
            return (p - p_k) * sqrt_AB
        else:  # rarefaction (isentrope)
            return g4 * c_k * ((p / p_k) ** g7 - 1.0)

    def _f_deriv(p, p_k, rho_k, c_k):
        if p > p_k:  # shock branch
            A = g5 / rho_k
            B = g6 * p_k
            sqrt_AB = np.sqrt(A / (p + B))
            return sqrt_AB * (1.0 - (p - p_k) / (2.0 * (p + B)))
        else:  # rarefaction branch
            return (1.0 / (rho_k * c_k)) * (p / p_k) ** (-gp1 / (2.0 * gamma))

    # ------------------------------------------------------------------ #
    #  Initial guess for p*  — Toro eq. (4.47):                          #
    #  "Two-Rarefaction" (TR) approximation                               #
    # ------------------------------------------------------------------ #
    p_TR = (
                   (c_l + c_r - 0.5 * gm1 * (u_r - u_l))
                   / (c_l / p_l ** g7 + c_r / p_r ** g7)
           ) ** (1.0 / g7)

    p0 = max(tol, p_TR)

    # ------------------------------------------------------------------ #
    #  Newton–Raphson iteration  — Toro eq. (4.44)                       #
    # ------------------------------------------------------------------ #
    p_new = p0
    for iteration in range(max_iter):
        p_cur = p_new
        fl = _f(p_cur, p_l, n_l, c_l)
        fr = _f(p_cur, p_r, n_r, c_r)
        fld = _f_deriv(p_cur, p_l, n_l, c_l)
        frd = _f_deriv(p_cur, p_r, n_r, c_r)

        p_new = p_cur - (fl + fr + (u_r - u_l)) / (fld + frd)
        p_new = max(p_new, tol)  # pressure must stay positive

        # Toro eq. (4.45): relative change criterion
        if 2.0 * abs(p_new - p_cur) / (p_new + p_cur) < tol:
            break
    else:
        raise RuntimeError(
            f"Newton iteration did not converge in {max_iter} iterations."
        )

    p_star = p_new
    u_star = 0.5 * (u_l + u_r) + 0.5 * (
            _f(p_star, p_r, n_r, c_r) - _f(p_star, p_l, n_l, c_l)
    )

    # ------------------------------------------------------------------ #
    #  Star-state densities  — Toro eq. (4.50) / (4.57)                  #
    #                                                                      #
    #  Isentrope:  n* = n_k * (p*/p_k)^(1/gamma)                         #
    #  Hugoniot:   n* = n_k * [ (gp1)*ratio + gm1 ]                      #
    #                        / [ gm1*ratio  + gp1 ]                      #
    #              ratio = p*/p_k                                          #
    # ------------------------------------------------------------------ #
    def _n_star(n_k, p_k, p_s):
        ratio = p_s / p_k
        if p_s <= p_k:  # rarefaction
            return n_k * ratio ** (1.0 / gamma)
        else:  # shock
            return n_k * (gp1 * ratio + gm1) / (gm1 * ratio + gp1)

    n_star_l = _n_star(n_l, p_l, p_star)
    n_star_r = _n_star(n_r, p_r, p_star)

    # ------------------------------------------------------------------ #
    #  Wave speeds  — Toro eq. (4.52) / (4.55)                           #
    #                                                                      #
    #  Shock speed:       S = u_k +/- c_k * sqrt((gp1*ratio+gm1)/(2g))  #
    #  Rarefaction speed: head = u_k +/- c_k                              #
    #                     tail = u* +/- c*                                #
    # ------------------------------------------------------------------ #
    c_star_l = np.sqrt(gamma * p_star / n_star_l)
    c_star_r = np.sqrt(gamma * p_star / n_star_r)

    if p_star <= p_l:  # left rarefaction
        S_l_head = u_l - c_l
        S_l_tail = u_star - c_star_l
    else:  # left shock
        S_l = u_l - c_l * np.sqrt((gp1 * (p_star / p_l) + gm1) / (2.0 * gamma))
        S_l_head = S_l_tail = S_l

    S_contact = u_star

    if p_star >= p_r:  # right shock
        S_r = u_r + c_r * np.sqrt((gp1 * (p_star / p_r) + gm1) / (2.0 * gamma))
        S_r_head = S_r_tail = S_r
    else:  # right rarefaction
        S_r_head = u_star + c_star_r
        S_r_tail = u_r + c_r

    # ------------------------------------------------------------------ #
    #  Sample solution on grid  — Toro section 4.5                        #
    #  Similarity variable: xi = (x - x0) / t                            #
    # ------------------------------------------------------------------ #
    xi = (x - x0) / t
    n_sol = np.empty_like(x)
    u_sol = np.empty_like(x)
    p_sol = np.empty_like(x)

    for i, s in enumerate(xi):
        if s <= S_l_head:
            # Region 1: undisturbed left state
            n_sol[i], u_sol[i], p_sol[i] = n_l, u_l, p_l

        elif s <= S_l_tail:
            # Inside left rarefaction fan  — Toro eq. (4.56)
            c_fan = g5 * (c_l + 0.5 * gm1 * (u_l - s))
            n_sol[i] = n_l * (c_fan / c_l) ** g4
            u_sol[i] = g5 * (c_l + 0.5 * gm1 * u_l + s)
            p_sol[i] = p_l * (c_fan / c_l) ** g3

        elif s <= S_contact:
            # Region 2: left star state
            n_sol[i], u_sol[i], p_sol[i] = n_star_l, u_star, p_star

        elif s <= S_r_head:
            # Region 3: right star state
            n_sol[i], u_sol[i], p_sol[i] = n_star_r, u_star, p_star

        elif s <= S_r_tail:
            # Inside right rarefaction fan  — Toro eq. (4.63)
            c_fan = g5 * (c_r - 0.5 * gm1 * (u_r - s))
            n_sol[i] = n_r * (c_fan / c_r) ** g4
            u_sol[i] = g5 * (-c_r + 0.5 * gm1 * u_r + s)
            p_sol[i] = p_r * (c_fan / c_r) ** g3

        else:
            # Region 4: undisturbed right state
            n_sol[i], u_sol[i], p_sol[i] = n_r, u_r, p_r

    return n_sol, u_sol, p_sol / n_sol  # n, u, T

def L2(x: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> float:
    dx = np.empty_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / 2.0  # внутренние точки
    dx[0]    = x[1] - x[0]              # левая граница
    dx[-1]   = x[-1] - x[-2]            # правая граница

    diff = f1 - f2
    return np.sqrt(np.sum(diff ** 2 * dx))

def L1(x: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> float:
    dx = np.empty_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / 2.0  # внутренние точки
    dx[0]    = x[1] - x[0]              # левая граница
    dx[-1]   = x[-1] - x[-2]            # правая граница

    diff = f1 - f2
    return np.sum(np.abs(diff) * dx)

def L_sup(x: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> float:
    return np.max(np.abs(f1 - f2))