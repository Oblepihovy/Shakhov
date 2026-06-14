from src.config.libloader import xp
from src.thermodynamics.model_state import ModelState
from src.thermodynamics.model_properties import ModelProperties
from src.thermodynamics.property_calculator import PropertyCalculator


class ShakhovSolver:

    def __init__(self, state: ModelState, properties: ModelProperties,
                 solver, prop_calc=PropertyCalculator):
        self.state = state
        self.props = properties
        self.solver = solver
        self.prop_calc = prop_calc

    def calculate(self, CFL, t_max):
        t_cur = 0
        n = 1
        xi_max = xp.max(xp.abs(self.props.xi))
        while t_cur < t_max:
            print(f"calculation: {t_cur} / {t_max}")
            tau = min(CFL * min(self.props.mesh.get_dx()) / xi_max,
                      max(t_max - t_cur, 1e-15))
            self.solver.calculate_layer(self.state.F, t_cur, tau, self.props, self.prop_calc)
            if n % 20 == 0:
                self.props.mesh.update(self.state.F[1:-1], self.prop_calc, self.props)
            t_cur += tau
            n += 1
