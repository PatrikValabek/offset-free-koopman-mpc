from scipy.integrate import solve_ivp
import numpy as np


class Model:
    def __init__(self, model_name = "Generic Model"):
        self.model_name = model_name
        
    def describe(self):
        print("Model Name: ", self.model_name)
    
    def log_state(self, state: dict):
        self.history.append(state.copy())
        
    def get_history(self):
        '''
        Returns the history of the model as a dictionary of vertical vectors
        '''
        keys = self.history[0].keys()
        result = {
            key: np.vstack([np.array(d[key]) if hasattr(d[key], '__len__') and not isinstance(d[key], str) else [d[key]] for d in self.history])
            for key in keys
        }

        return result
    
    def reset_history(self):
        self.history = []
    
    def simulate(self, y0: list, u:list, Ts: float) -> list:
        self.reset_history()
        self.u_data = u
        self.Ts = Ts
            
        state = {"Y": y0, "U": u[0,:]}
            
        for i in range(u.shape[0]):
            t_span = [i * Ts, (i + 1) * Ts]
            sol = solve_ivp(self.ode, t_span, y0, t_eval=t_span, method='RK45')
            y0 = sol.y[:,-1]
            state = {"Y": y0, "U": u[i,:]}
            self.log_state(state)
        
        return self.get_history()
        
    