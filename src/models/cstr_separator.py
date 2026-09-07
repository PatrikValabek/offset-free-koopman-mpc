from .model import Model
import numpy as np
from typing import List, Tuple


class CSTRSeparator(Model):
    """
    Two CSTRs plus flash separator with recycle (Liu et al. 2009 structure,
    Li & Swartz 2019 parameters).

    References:
        Liu, J., Muñoz de la Peña, D., & Christofides, P. D. (2009).
            Distributed Model Predictive Control of Nonlinear Process Systems.
            AIChE Journal, 55(5), 1171-1184.
        Li, H., & Swartz, C. L. E. (2019). Computers & Chemical Engineering.

    Reactions: A -> B -> C (A reactant, B desired product, C undesired).
    Constant vessel inventory: F1 = F10 + Fr, F2 = F1 + F20, Fp = F10 + F20.

    States (9):
        xA1, xB1, T1, xA2, xB2, T2, xA3, xB3, T3
        - xAi, xBi: mass fractions of A and B in vessel i
        - Ti: temperature in vessel i [K]
        Vessel 3 is the flash separator (no reaction).

    Inputs (6): [Q1, Q2, Q3, F10, F20, Fr]
        - Q1, Q2, Q3: heat inputs [kJ/s]
        - F10: feed to vessel 1 [m³/s]
        - F20: feed to vessel 2 [m³/s]
        - Fr: recycle from separator to vessel 1 [m³/s]

    Typical measurements: [T1, T2, T3, xB3] (separator bottoms quality).

    Nominal steady inputs (Li & Swartz Table 5):
        Q1=Q2=Q3=10 kJ/s, F10=8.3 m³/s, F20=0.5 m³/s, Fr=4.0 m³/s
    """

    def __init__(
        self,
        # Kinetic / physical (Li & Swartz Table 4)
        rho: float = 0.15,  # kg/m³
        E1: float = 813.4,  # J/mol
        E2: float = 1247.1,  # J/mol
        T10: float = 313.0,  # K
        T20: float = 313.0,  # K
        V1: float = 89.4,  # m³
        V2: float = 90.0,  # m³
        V3: float = 13.27,  # m³
        k1: float = 0.336,  # 1/s
        k2: float = 0.089,  # 1/s
        R: float = 8.314,  # J/(mol·K)
        xA10: float = 1.0,
        xB10: float = 0.0,
        xA20: float = 1.0,
        xB20: float = 0.0,
        DH1: float = -40.0,  # kJ/kg
        DH2: float = -50.0,  # kJ/kg
        Cp: float = 2.5,  # kJ/(kg·K)
        alpha_A: float = 3.5,
        alpha_B: float = 0.5,
        alpha_C: float = 1.1,
    ):
        super().__init__(model_name="CSTR-Separator (Li & Swartz 2019)")

        self.rho = rho
        self.E1 = E1
        self.E2 = E2
        self.T10 = T10
        self.T20 = T20
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.k1 = k1
        self.k2 = k2
        self.R = R
        self.xA10 = xA10
        self.xB10 = xB10
        self.xA20 = xA20
        self.xB20 = xB20
        self.DH1 = DH1
        self.DH2 = DH2
        self.Cp = Cp
        self.alpha_A = alpha_A
        self.alpha_B = alpha_B
        self.alpha_C = alpha_C

        # Nominal plant input [Q1, Q2, Q3, F10, F20, Fr]
        self.u_nom = np.array([10.0, 10.0, 10.0, 8.3, 0.5, 4.0], dtype=float)
        # Wide identification / control box for MVs [Q1, Q2, Q3, F10].
        # Spans are large enough that random steps leave the Li–Swartz
        # neighborhood, so linear / local Koopman models drift at new OPs.
        self.mv_constraints = np.array(
            [
                [0.0, 25.0],  # Q1 [kJ/s]
                [0.0, 25.0],  # Q2 [kJ/s]
                [0.0, 25.0],  # Q3 [kJ/s]
                [3.0, 16.0],  # F10 [m³/s]
            ],
            dtype=float,
        )
        self.y_names = ["T1", "T2", "T3", "xB3"]
        self.u_names = ["Q1", "Q2", "Q3", "F10"]
        # Initial guess near Li Table 5
        self.x0_guess = np.array(
            [
                0.250329493015859,
                0.418357085601851,
                336.7454114199064,
                0.100657975048143,
                0.413596407210689,
                344.1119605116316,
                0.022164588403115,
                0.637508939407261,
                346.1952938449650,
            ],
            dtype=float,
        )

    def flash_recycle(self, xA3: float, xB3: float) -> Tuple[float, float]:
        """Relative-volatility flash (Liu Eq. 15). Returns (xAr, xBr)."""
        xC3 = max(1.0 - xA3 - xB3, 0.0)
        den = self.alpha_A * xA3 + self.alpha_B * xB3 + self.alpha_C * xC3
        if den > 1e-10:
            xAr = self.alpha_A * xA3 / den
            xBr = self.alpha_B * xB3 / den
        else:
            xAr = 0.0
            xBr = 0.0
        return xAr, xBr

    def measure(self, x: np.ndarray) -> np.ndarray:
        """Industry-like measurements: [T1, T2, T3, xB3]."""
        x = np.asarray(x, dtype=float).reshape(9)
        return np.array([x[2], x[5], x[8], x[7]], dtype=float)

    def ode(self, t: float, x: np.ndarray) -> List[float]:
        """
        ODEs of the two CSTRs and flash separator.

        x: [xA1, xB1, T1, xA2, xB2, T2, xA3, xB3, T3]
        u(t): [Q1, Q2, Q3, F10, F20, Fr]
        """
        xA1, xB1, T1, xA2, xB2, T2, xA3, xB3, T3 = x

        xA1 = max(xA1, 0.0)
        xB1 = max(xB1, 0.0)
        xA2 = max(xA2, 0.0)
        xB2 = max(xB2, 0.0)
        xA3 = max(xA3, 0.0)
        xB3 = max(xB3, 0.0)
        T1 = max(T1, 1.0)
        T2 = max(T2, 1.0)
        T3 = max(T3, 1.0)

        xAr, xBr = self.flash_recycle(xA3, xB3)

        u = self.get_input(t)
        Q1, Q2, Q3, F10, F20, Fr = u[0], u[1], u[2], u[3], u[4], u[5]

        F1 = F10 + Fr
        F2 = F1 + F20
        Fp = F10 + F20
        Fr_plus_Fp = Fr + Fp

        r1_1 = self.k1 * np.exp(-self.E1 / (self.R * T1)) * xA1
        r2_1 = self.k2 * np.exp(-self.E2 / (self.R * T1)) * xB1
        r1_2 = self.k1 * np.exp(-self.E1 / (self.R * T2)) * xA2
        r2_2 = self.k2 * np.exp(-self.E2 / (self.R * T2)) * xB2

        dxA1dt = (
            (F10 / self.V1) * (self.xA10 - xA1) + (Fr / self.V1) * (xAr - xA1) - r1_1
        )
        dxB1dt = (
            (F10 / self.V1) * (self.xB10 - xB1)
            + (Fr / self.V1) * (xBr - xB1)
            + r1_1
            - r2_1
        )
        dT1dt = (
            (F10 / self.V1) * (self.T10 - T1)
            + (Fr / self.V1) * (T3 - T1)
            + (-self.DH1 / self.Cp) * r1_1
            + (-self.DH2 / self.Cp) * r2_1
            + Q1 / (self.rho * self.Cp * self.V1)
        )

        dxA2dt = (
            (F1 / self.V2) * (xA1 - xA2) + (F20 / self.V2) * (self.xA20 - xA2) - r1_2
        )
        dxB2dt = (
            (F1 / self.V2) * (xB1 - xB2)
            + (F20 / self.V2) * (self.xB20 - xB2)
            + r1_2
            - r2_2
        )
        dT2dt = (
            (F1 / self.V2) * (T1 - T2)
            + (F20 / self.V2) * (self.T20 - T2)
            + (-self.DH1 / self.Cp) * r1_2
            + (-self.DH2 / self.Cp) * r2_2
            + Q2 / (self.rho * self.Cp * self.V2)
        )

        dxA3dt = (F2 / self.V3) * (xA2 - xA3) - (Fr_plus_Fp / self.V3) * (xAr - xA3)
        dxB3dt = (F2 / self.V3) * (xB2 - xB3) - (Fr_plus_Fp / self.V3) * (xBr - xB3)
        dT3dt = (F2 / self.V3) * (T2 - T3) + Q3 / (self.rho * self.Cp * self.V3)

        return [dxA1dt, dxB1dt, dT1dt, dxA2dt, dxB2dt, dT2dt, dxA3dt, dxB3dt, dT3dt]

    def get_input(self, t: float) -> np.ndarray:
        """
        Return the control input vector [Q1, Q2, Q3, F10, F20, Fr] at time t
        based on the stored `u_data` and sampling time `Ts`.
        """
        if t % self.Ts == 0 and t != 0:
            index = int(t // self.Ts) - 1
        else:
            index = int(t // self.Ts)

        index = min(index, self.u_data.shape[0] - 1)
        return self.u_data[index, :]
