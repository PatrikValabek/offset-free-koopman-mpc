from .model import Model
import numpy as np
from typing import List

class ReactorSeparator(Model):
    """
    Reactor-Separator System from Liu et al. (2009) AIChE Journal.
    
    Reference:
        Liu, J., Muñoz de la Peña, D., & Christofides, P. D. (2009). 
        Distributed Model Predictive Control of Nonlinear Process Systems. 
        AIChE Journal, 55(5), 1171-1184.
    
    System Description:
        A three vessel reactor-separator process consisting of two continuously
        stirred tank reactors (CSTRs) and a flash tank separator.
        
        Reactions: A -> B -> C
        where A is the reactant, B is the desired product, and C is an undesired side-product.
    
    States (9 total):
        xA1, xB1: Mass fractions of A and B in vessel 1
        T1: Temperature in vessel 1 [K]
        xA2, xB2: Mass fractions of A and B in vessel 2
        T2: Temperature in vessel 2 [K]
        xA3, xB3: Mass fractions of A and B in vessel 3 (separator)
        T3: Temperature in vessel 3 [K]
        
    Control inputs (6 total):
        F10: Feed stream flow rate to vessel 1 [m³/s]
        Q1: Heat input to vessel 1 [kJ/s]
        F20: Feed stream flow rate to vessel 2 [m³/s]
        Q2: Heat input to vessel 2 [kJ/s]
        Fr: Recycle flow rate from separator to vessel 1 [m³/s]
        Q3: Heat input to vessel 3 [kJ/s]
        
    Time Units:
        All equations are formulated with time in SECONDS:
        - Sampling time Ts should be in seconds
        - Flow rates F are in m³/s
        - Heat inputs Q are in kJ/s
        - Reaction rate constants k are in s⁻¹
        
    Note on Parameters:
        Parameters are taken from Table 4 of Liu et al. (2009).
        
    Steady-State from Paper (Table 5):
        States: xA1s=0.264, xB1s=0.396, T1s=337.02K, xA2s=0.106, xB2s=0.404, 
                T2s=344.43K, xA3s=0.057, xB3s=0.475, T3s=346.51K
        Inputs: F10s=8.3 m³/s, Q1s=10 kJ/s, F20s=0.5 m³/s, Q2s=10 kJ/s, 
                Frs=4 m³/s, Q3s=10 kJ/s
    """
    
    def __init__(
        self,
        # Vessel volumes [m³] - from paper table
        V1: float = 89.4,
        V2: float = 90.0,
        V3: float = 13.27,
        # Pre-exponential factors [1/s] - from paper table
        k1: float = 0.336,  # s⁻¹
        k2: float = 0.089,  # s⁻¹
        # Activation energies [J/mol] - from paper table
        E1: float = 813.4,  # J/mol
        E2: float = 1247.1,    # J/mol
        # Heats of reaction [kJ/kg] - from paper table
        DH1: float = -40.0,   # kJ/kg
        DH2: float = -50.0,   # kJ/kg
        # Physical properties - from paper table
        Cp: float = 2.5,      # Heat capacity [kJ/(kg·K)]
        rho: float = 915.0,   # Density [kg/m³]
        R: float = 8.314,     # Gas constant [J/(mol·K)]
        # Relative volatilities - from paper table
        alpha_A: float = 3.5,
        alpha_B: float = 0.5,
        alpha_C: float = 1.1,
        # Purge flow rate [m³/s] - fixed parameter
        Fp: float = 0.5,  # Purge flow rate (from paper, typically ~10% of recycle)
        # Feed stream properties - from paper table
        xA10: float = 1.0,    # Pure A in feed 1
        xB10: float = 0.0,
        T10: float = 313.0,   # [K] - from paper table (T30)
        xA20: float = 1.0,    # Pure A in feed 2
        xB20: float = 0.0,
        T20: float = 313.0,   # [K] - from paper table (T30)
    ):
        super().__init__(model_name="Reactor-Separator System (Liu 2009)")
        
        # Store parameters
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.k1 = k1
        self.k2 = k2
        self.E1 = E1
        self.E2 = E2
        self.DH1 = DH1
        self.DH2 = DH2
        self.Cp = Cp
        self.rho = rho
        self.R = R
        self.alpha_A = alpha_A
        self.alpha_B = alpha_B
        self.alpha_C = alpha_C
        self.Fp = Fp
        self.xA10 = xA10
        self.xB10 = xB10
        self.T10 = T10
        self.xA20 = xA20
        self.xB20 = xB20
        self.T20 = T20
        
    def ode(self, t: float, x: np.ndarray) -> List[float]:
        """
        Differential equations for the reactor-separator system.
        
        Parameters:
        -----------
        t : float
            Current time
        x : np.ndarray
            State vector [xA1, xB1, T1, xA2, xB2, T2, xA3, xB3, T3]
            
        Returns:
        --------
        list[float]
            Time derivatives of state variables
        """
        # Unpack states
        xA1, xB1, T1, xA2, xB2, T2, xA3, xB3, T3 = x
        
        # Prevent negative mass fractions and ensure temperatures are reasonable
        xA1 = max(xA1, 0.0)
        xB1 = max(xB1, 0.0)
        xA2 = max(xA2, 0.0)
        xB2 = max(xB2, 0.0)
        xA3 = max(xA3, 0.0)
        xB3 = max(xB3, 0.0)
        T1 = max(T1, 1.0)  # Prevent division by zero in reaction rates
        T2 = max(T2, 1.0)
        T3 = max(T3, 1.0)
        
        # Calculate xC3 from mass balance
        xC3 = max(1.0 - xA3 - xB3, 0.0)
        
        # Calculate recycle compositions using algebraic equations (flash separator)
        denominator = self.alpha_A * xA3 + self.alpha_B * xB3 + self.alpha_C * xC3
        if denominator > 1e-10:
            xAr = self.alpha_A * xA3 / denominator
            xBr = self.alpha_B * xB3 / denominator
        else:
            xAr = 0.0
            xBr = 0.0
        
        # Get control inputs at current time
        u = self.get_input(t)
        F10, Q1, F20, Q2, Fr, Q3 = u[0], u[1], u[2], u[3], u[4], u[5]
        
        # Calculate flow rates between vessels
        # F1: flow from vessel 1 to vessel 2 = F10 + Fr
        # F2: flow from vessel 2 to vessel 3 = F1 + F20
        F1 = F10 + Fr
        F2 = F1 + F20
        
        # Reaction rates using Arrhenius equation
        # k is in s⁻¹, no conversion needed since time unit is seconds
        r1_1 = self.k1 * np.exp(-self.E1 / (self.R * T1)) * xA1
        r2_1 = self.k2 * np.exp(-self.E2 / (self.R * T1)) * xB1
        r1_2 = self.k1 * np.exp(-self.E1 / (self.R * T2)) * xA2
        r2_2 = self.k2 * np.exp(-self.E2 / (self.R * T2)) * xB2
        
        # Differential equations for CSTR 1
        dxA1dt = (F10 / self.V1) * (self.xA10 - xA1) + \
                 (Fr / self.V1) * (xAr - xA1) - r1_1
        
        dxB1dt = (F10 / self.V1) * (self.xB10 - xB1) + \
                 (Fr / self.V1) * (xBr - xB1) + r1_1 - r2_1
        
        dT1dt = (F10 / self.V1) * (self.T10 - T1) + \
                (Fr / self.V1) * (T3 - T1) + \
                (-self.DH1 / self.Cp) * r1_1 + \
                (-self.DH2 / self.Cp) * r2_1 + \
                Q1 / (self.rho * self.Cp * self.V1)
        
        # Differential equations for CSTR 2
        dxA2dt = (F1 / self.V2) * (xA1 - xA2) + \
                 (F20 / self.V2) * (self.xA20 - xA2) - r1_2
        
        dxB2dt = (F1 / self.V2) * (xB1 - xB2) + \
                 (F20 / self.V2) * (self.xB20 - xB2) + r1_2 - r2_2
        
        dT2dt = (F1 / self.V2) * (T1 - T2) + \
                (F20 / self.V2) * (self.T20 - T2) + \
                (-self.DH1 / self.Cp) * r1_2 + \
                (-self.DH2 / self.Cp) * r2_2 + \
                Q2 / (self.rho * self.Cp * self.V2)
        
        # Differential equations for Separator (Flash Tank)
        dxA3dt = (F2 / self.V3) * (xA2 - xA3) - \
                 ((Fr + self.Fp) / self.V3) * (xAr - xA3)
        
        dxB3dt = (F2 / self.V3) * (xB2 - xB3) - \
                 ((Fr + self.Fp) / self.V3) * (xBr - xB3)
        
        dT3dt = (F2 / self.V3) * (T2 - T3) + \
                Q3 / (self.rho * self.Cp * self.V3)
        
        return [dxA1dt, dxB1dt, dT1dt, dxA2dt, dxB2dt, dT2dt, dxA3dt, dxB3dt, dT3dt]
    
    def get_input(self, t: float) -> np.ndarray:
        """
        Get control input at time t.
        
        Parameters:
        -----------
        t : float
            Current time
            
        Returns:
        --------
        np.ndarray
            Control input vector [F10, Q1, F20, Q2, Fr, Q3] at time t
        """
        if t % self.Ts == 0 and t != 0:
            index = int(t // self.Ts) - 1
        else:
            index = int(t // self.Ts)
        
        index = min(index, self.u_data.shape[0] - 1)
        return self.u_data[index, :]
