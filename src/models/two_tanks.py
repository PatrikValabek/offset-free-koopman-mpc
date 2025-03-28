from .model import Model
import numpy as np

class TwoTanks(Model):
    def __init__(
        self,
        interaction: bool = True, 
        A1: float=1, 
        A2: float=1, 
        k1: float=0.5, 
        k2: float=0.3
    ):
        if interaction:
            super().__init__(model_name="Two Tank System with Interaction")
        elif not interaction:
            super().__init__(model_name="Two Tank System without Interaction")
        else:
            raise ValueError("interaction must be True or False")
        self.interaction = interaction
        self.A1 = A1
        self.A2 = A2
        self.k1 = k1
        self.k2 = k2
        
    def ode(self, t: float, h: np.ndarray) -> list[float]:
        h1, h2 = h
        h1 = max(h1, 0.0)  # prevent negative heights
        h2 = max(h2, 0.0)

        if self.interaction:
            delta_h = h1 - h2
            flow12 = self.k1 * np.sign(delta_h) * np.sqrt(abs(delta_h))
        else:
            flow12 = 0.0

        outflow = self.k2 * np.sqrt(h2)

        dh1dt = (self.u1(t) - flow12) / self.A1
        dh2dt = (flow12 + self.u2(t) - outflow) / self.A2

        return [dh1dt, dh2dt]
    
    def u1(self, t: float) -> float:
        if t % self.Ts == 0 and t != 0:
            index = int(t // self.Ts) - 1
        else:
            index = int(t // self.Ts)
        return self.u_data[index, 0]
    
    def u2(self, t: float) -> float:
        if t % self.Ts == 0 and t != 0:
            index = int(t // self.Ts) - 1
        else:
            index = int(t // self.Ts)
        return self.u_data[index, 1]
    
    # def simulate(self, h1: float, h2: float, u1: list, u2: list, Ts: float) -> list:
    #     self.reset_history()
    #     self.u1_data = u1
    #     self.u2_data = u2
    #     self.Ts = Ts
            
    #     state = {"h1": h1, "h2": h2, "u1": u1[0], "u2": u2[0]}
            
    #     for i in range(len(u1)):
    #         t_span = [i * Ts, (i + 1) * Ts]
    #         sol = solve_ivp(self.ode, t_span, [h1,h2], t_eval=t_span, method='RK45')
    #         h1 = sol.y[0][-1]
    #         h2 = sol.y[1][-1]
    #         state = {"h1": h1, "h2": h2, "u1": u1[i], "u2": u2[i]}
    #         self.log_state(state)
            
    #     return self.get_history()
    

        
