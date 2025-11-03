from .model import Model
import numpy as np
from typing import List


class CSTRSeriesRecycle(Model):
    """
    Two CSTRs in series with a recycle loop and jacket cooling for each tank.

    Chemistry: Parallel reactions on reactant A (classic selectivity case)
        Desired:   2 A -> B   with rate r1 = k1(T) * C_A^2   [second-order in A]
        Undesired: A  -> U   with rate r2 = k2(T) * C_A      [first-order in A]
        Shunt (R1 only): A + B -> 2 B with rate r3 = k3(T) * C_A * C_B

    States (8): [C_A1, T1, C_A2, T2, C_B1, C_B2, C_U1, C_U2]
        - C_A1, C_A2: reactant A concentrations [mol/m^3]
        - C_B1, C_B2: desired product B concentrations [mol/m^3]
        - C_U1, C_U2: undesired product U concentrations [mol/m^3]
        - T1, T2: temperatures [K]

    Inputs (4): [F, L, Tc1, Tc2]
        - F: fresh feed flow rate [m^3/s]
        - L: recycle flow rate [m^3/s]
        - Tc1, Tc2: coolant (jacket) temperatures [K]

    Notes:
        - Arrhenius temperature dependence: k_i(T) = k_i0 * exp(-E_i / (R * T))
        - Heat release terms use (-deltaH_i) with deltaH_i > 0 for exothermic reactions.
        - Feed contains only A by default (B and U feeds are zero).
    """

    def __init__(
        self,
        C_A_O: float = 97.35,   # mol/m^3, feed concentration of A (C_A,0)
        T_O: float = 298.0,   # K, feed temperature
        C_B_O: float = 0.0,   # mol/m^3, feed concentration of B
        C_U_O: float = 0.0,   # mol/m^3, feed concentration of U
        V1: float = 1e-3,     # m^3
        V2: float = 2e-3,     # m^3
        U1A1: float = 0.461,  # kJ/(s·K)
        U2A2: float = 0.732,  # kJ/(s·K)
        rho: float = 1.05e3,  # kg/m^3
        cp: float = 3.766,    # kJ/(kg·K)
        # Kinetics (Arrhenius parameters)
        k1_0: float = 1.0e5,   # m^3/(mol·s), pre-exponential for r1 (2A->B)
        E1: float = 45,     # kJ/mol, activation energy for r1
        k2_0: float = 9.8e9, # 1/s, pre-exponential for r2 (A->U)
        E2: float = 70,     # kJ/mol, activation energy for r2
        # Autocatalytic shunt (R1 only): A + B -> 2 B
        k3_0: float = 5.0e4,  # m^3/(mol·s), pre-exponential for r3 (A+B->2B)
        E3: float = 55,       # kJ/mol, activation energy for r3
        # Heats of reaction (positive magnitude; exothermic heat = -deltaH)
        deltaH1: float = 60,  # kJ/mol, heat for 2A->B based on extent of r1
        deltaH2: float = 40,  # kJ/mol, heat for A->U based on extent of r2
        deltaH3: float = 60,  # kJ/mol, heat for A+B->2B based on extent of r3
        R: float = 8.3145e-3,   # kJ/(mol·K)
    ):
        super().__init__(model_name="CSTR Series with Recycle")

        # Store parameters
        self.C_A_O = C_A_O
        self.T_O = T_O
        self.C_B_O = C_B_O
        self.C_U_O = C_U_O
        self.V1 = V1
        self.V2 = V2
        self.U1A1 = U1A1
        self.U2A2 = U2A2
        self.rho = rho
        self.cp = cp
        self.k1_0 = k1_0
        self.k2_0 = k2_0
        self.E1 = E1
        self.E2 = E2
        self.k3_0 = k3_0
        self.E3 = E3
        self.deltaH1 = deltaH1
        self.deltaH2 = deltaH2
        self.deltaH3 = deltaH3
        self.R = R

    def ode(self, t: float, x: np.ndarray) -> List[float]:
        """
        ODEs of the two CSTRs with recycle and parallel reactions.

        x: [C_A1, T1, C_A2, T2, C_B1, C_B2, C_U1, C_U2]
        u(t): [F, L, Tc1, Tc2]
        """
        C_A1, T1, C_A2, T2, C_B1, C_B2, C_U1, C_U2 = x

        # Guard against non-physical states
        C_A1 = max(C_A1, 0.0)
        C_A2 = max(C_A2, 0.0)
        C_B1 = max(C_B1, 0.0)
        C_B2 = max(C_B2, 0.0)
        C_U1 = max(C_U1, 0.0)
        C_U2 = max(C_U2, 0.0)
        T1 = max(T1, 1.0)
        T2 = max(T2, 1.0)

        u = self.get_input(t)
        F, L, Tc1, Tc2 = u[0], u[1], u[2], u[3]

        # Arrhenius terms
        k1_T1 = self.k1_0 * np.exp(-self.E1 / (self.R * T1))
        k1_T2 = self.k1_0 * np.exp(-self.E1 / (self.R * T2))
        k2_T1 = self.k2_0 * np.exp(-self.E2 / (self.R * T1))
        k2_T2 = self.k2_0 * np.exp(-self.E2 / (self.R * T2))
        k3_T1 = self.k3_0 * np.exp(-self.E3 / (self.R * T1))  # shunt active only in R1

        # Reaction rates
        r1_1 = k1_T1 * (C_A1 ** 2)   # 2A -> B (second order in A)
        r2_1 = k2_T1 * C_A1          # A  -> U (first order in A)
        r3_1 = k3_T1 * C_A1 * C_B1   # A + B -> 2B (autocatalytic shunt in R1)
        r1_2 = k1_T2 * (C_A2 ** 2)
        r2_2 = k2_T2 * C_A2

        # Reactor 1 balances
        dCA1dt = (self.C_A_O / self.V1) * F + (L / self.V1) * C_A2 - ((F + L) / self.V1) * C_A1 \
                 - 2.0 * r1_1 - r2_1 - r3_1
        dCB1dt = (self.C_B_O / self.V1) * F + (L / self.V1) * C_B2 - ((F + L) / self.V1) * C_B1 \
                 + r1_1 + r3_1  # net +1 B from A+B->2B
        dCU1dt = (self.C_U_O / self.V1) * F + (L / self.V1) * C_U2 - ((F + L) / self.V1) * C_U1 \
                 + r2_1
        dT1dt = (self.T_O / self.V1) * F + (L / self.V1) * T2 \
                - (self.U1A1 / (self.V1 * self.rho * self.cp)) * (T1 - Tc1) \
                - ((F + L) / self.V1) * T1 \
                + ((-self.deltaH1) * r1_1 + (-self.deltaH2) * r2_1 + (-self.deltaH3) * r3_1) / (self.rho * self.cp)

        # Reactor 2 balances (no fresh feed; receives R1 outlet)
        dCA2dt = ((F + L) / self.V2) * (C_A1 - C_A2) - 2.0 * r1_2 - r2_2
        dCB2dt = ((F + L) / self.V2) * (C_B1 - C_B2) + r1_2
        dCU2dt = ((F + L) / self.V2) * (C_U1 - C_U2) + r2_2
        dT2dt = ((F + L) / self.V2) * (T1 - T2) \
                - (self.U2A2 / (self.V2 * self.rho * self.cp)) * (T2 - Tc2) \
                + ((-self.deltaH1) * r1_2 + (-self.deltaH2) * r2_2) / (self.rho * self.cp)

        return [dCA1dt, dT1dt, dCA2dt, dT2dt, dCB1dt, dCB2dt, dCU1dt, dCU2dt]

    def get_input(self, t: float) -> np.ndarray:
        """
        Return the control input vector [F, L, Tc1, Tc2] at time t
        based on the stored `u_data` and sampling time `Ts`.
        """
        if t % self.Ts == 0 and t != 0:
            index = int(t // self.Ts) - 1
        else:
            index = int(t // self.Ts)

        index = min(index, self.u_data.shape[0] - 1)
        return self.u_data[index, :]