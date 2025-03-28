import numpy as np
import torch

from .koopman import evaluate_jacobian

class EKF():
    
    def __init__(self, A, B, x0, P0, problem, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.problem = problem
    
    def get_y(self, x):
        y = self.problem.nodes[4]({"x": torch.from_numpy(x.T).float()})
        return y["yhat"].detach().numpy().reshape(1,-1)
    
    def predict(self, u):
        self.x = self.A @ self.x.reshape(-1,1) + self.B @ u.reshape(-1,1)
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x, self.P
    
    def update(self, y):
        H = evaluate_jacobian(self.problem.nodes[4], torch.tensor(self.x).T[0])
        y_pred = self.get_y(self.x.T[0])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - y_pred).T
        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P
        
    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.x
    
    
    
        