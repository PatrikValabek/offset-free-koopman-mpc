import numpy as np
from numpy.linalg import inv
import torch

from .koopman import evaluate_jacobian

class KF():
    def __init__(self, A, B, C, x0, P0, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0  
        
    def predict(self, u):
        self.x = self.A @ self.x.reshape(-1,1) + self.B @ u.reshape(-1,1)
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x, self.P
    
    def update(self, y):
        y_pred = self.C@self.x.reshape(-1,1)
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - y_pred.T).T
        self.P = (np.eye(self.P.shape[0]) - K @ self.C) @ self.P
        
    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.x
        
    
class EKF():
    def __init__(self, A, B, x0, P0, problem, Q, R, disturbance=0, T_real = None):
        self.disturbance = disturbance
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.problem = problem
        if T_real is not None:
            self.T_real = T_real
        else:
            self.T_real = np.eye(A.shape[0]-disturbance)
        self.nd = disturbance
        self.nx = A.shape[0] - self.nd
    
    def get_y(self, x):
        y = self.problem.nodes[4]({"x": torch.from_numpy(x.T).float()})
        return y["yhat"].detach().numpy().reshape(1,-1)
    
    def predict(self, u):
        self.x = self.A @ self.x.reshape(-1,1) + self.B @ u.reshape(-1,1)
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x, self.P
    
    def update(self, y):
        y_pred = self.get_y(self.T_real@self.x.T[0,0:self.nx]) + self.x.T[0,self.nx:]
        J = evaluate_jacobian(self.problem.nodes[4], torch.tensor(self.x[:self.nx]).T[0])
        self.H = np.hstack([J@self.T_real, np.eye(self.nd)])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - y_pred).T
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        
    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.x
    
class EKF_test():
    
    def __init__(self, A, B, x0, P0, problem, Q, R, disturbance=False, F=None, C=None):
        self.disturbance = disturbance
        self.problem = problem
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

        if disturbance:
            assert F is not None and C is not None, "F and C must be provided if disturbance is True"

            self.F = F
            self.C = C

            nx = A.shape[0]
            nd = F.shape[1]
            nu = B.shape[1]

            # Augmented dynamics
            self.A = np.block([
                [A, np.zeros((nx, nd))],
                [np.zeros((nd, nx)), np.eye(nd)]
            ])
            self.B = np.vstack([
                B,
                np.zeros((nd, nu))
            ])
        else:
            self.A = A
            self.B = B
            self.C = C
            self.F = np.zeros((C.shape[0], 0))  # no disturbance part

    
    def get_y(self, x_physical, d=None):
        """
        Get predicted measurement y from physical state x and optional disturbance d
        """
        yhat = self.C @ x_physical.reshape(-1, 1)
        if self.disturbance and d is not None:
            yhat = self.C @ x_physical.reshape(-1, 1) +  self.F @ d.reshape(-1, 1)

        return yhat.reshape(1, -1)
    
    def predict(self, u):
        self.x = self.A @ self.x.reshape(-1, 1) + self.B @ u.reshape(-1, 1)
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x, self.P
    
    def update(self, y):
        nx = self.C.shape[1]  # physical state dimension
        x_phys = self.x[:nx]           # only physical state
        d_est = self.x[nx:] if self.disturbance else None

        H = evaluate_jacobian(self.problem.nodes[4], torch.tensor(x_phys).T[0])
        y_pred = self.get_y(x_phys, d_est)

        H_aug = np.hstack([H, self.F])  # Total measurement Jacobian wrt [x; d]
        S = H_aug @ self.P @ H_aug.T + self.R
        K = self.P @ H_aug.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - y_pred).T
        self.P = (np.eye(self.P.shape[0]) - K @ H_aug) @ self.P


    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.x
    
class EKF_noC():
    
    def __init__(self, A, B, x0, P0, problem, Q, R, disturbance=False, F=None, C=None):
        self.disturbance = disturbance
        self.problem = problem
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

        if disturbance:
            assert F is not None and C is not None, "F and C must be provided if disturbance is True"

            self.F = F
            self.C = C

            nx = A.shape[0]
            nd = F.shape[1]
            nu = B.shape[1]

            # Augmented dynamics
            self.A = np.block([
                [A, np.zeros((nx, nd))],
                [np.zeros((nd, nx)), np.eye(nd)]
            ])
            self.B = np.vstack([
                B,
                np.zeros((nd, nu))
            ])
        else:
            self.A = A
            self.B = B
            self.C = C
            self.F = np.zeros((C.shape[0], 0))  # no disturbance part

    
    def get_y(self, x_physical, d=None):
        """
        Get predicted measurement y from physical state x and optional disturbance d
        """
        y = self.problem.nodes[4]({"x": torch.from_numpy(x_physical.T).float()})
        yhat = y["yhat"].detach().numpy().reshape(-1, 1)
        if self.disturbance and d is not None:
            yhat = yhat +  self.F @ d.reshape(-1, 1)

        return yhat.reshape(1, -1)
    
    def predict(self, u):
        self.x = self.A @ self.x.reshape(-1, 1) + self.B @ u.reshape(-1, 1)
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x, self.P
    
    def update(self, y):
        nx = self.C.shape[1]  # physical state dimension
        x_phys = self.x[:nx]           # only physical state
        d_est = self.x[nx:] if self.disturbance else None

        H = evaluate_jacobian(self.problem.nodes[4], torch.tensor(x_phys).T[0])
        y_pred = self.get_y(x_phys, d_est)

        H_aug = np.hstack([H, self.F])  # Total measurement Jacobian wrt [x; d]
        S = H_aug @ self.P @ H_aug.T + self.R
        K = self.P @ H_aug.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - y_pred).T
        self.P = (np.eye(self.P.shape[0]) - K @ H_aug) @ self.P


    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.x


    
    
        