import numpy as np
from numpy.linalg import inv
import torch
import cvxpy as cp
import joblib

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
        J = evaluate_jacobian(self.problem.nodes[4], torch.tensor(self.T_real@self.x[:self.nx]).T[0])
        self.H = np.hstack([J@self.T_real, np.eye(self.nd)])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - y_pred).T
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        
    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.x
    

class UKF():
    def __init__(self, A, B, x0, P0, problem, Q, R, disturbance=0, T_real=None,
                 alpha=1e-3, beta=2, kappa=0):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.problem = problem
        self.nd = disturbance
        self.nx = A.shape[0] - self.nd

        if T_real is not None:
            self.T_real = T_real
        else:
            self.T_real = np.eye(self.nx)

        self.n = A.shape[0]  # full state dimension (including disturbance if any)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = alpha**2 * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lam)

        # Weights for mean and covariance
        self.Wm = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lam)))
        self.Wc = np.copy(self.Wm)
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

    def get_y(self, x):
        """Measurement model h(x) using the NN node from problem.nodes[4]."""
        y = self.problem.nodes[4]({"x": torch.from_numpy(x.T).float()})
        return y["yhat"].detach().numpy().reshape(1, -1)

    def f(self, x, u):
        """State transition model: linear dynamics."""
        return self.A @ x.reshape(-1, 1) + self.B @ u.reshape(-1, 1)

    def generate_sigma_points(self):
        """Generates 2n+1 sigma points."""
        sigma_pts = [self.x.flatten()]
        sqrt_P = np.linalg.cholesky((self.n + self.lam) * self.P)
        for i in range(self.n):
            sigma_pts.append((self.x.flatten() + sqrt_P[:, i]))
            sigma_pts.append((self.x.flatten() - sqrt_P[:, i]))
        return np.array(sigma_pts)

    def predict(self, u):
        sigma_pts = self.generate_sigma_points()
        X_pred = np.array([self.f(x, u).flatten() for x in sigma_pts])
        self.x = np.sum(self.Wm[:, None] * X_pred, axis=0).reshape(-1, 1)

        self.P = self.Q.copy()
        for i in range(2 * self.n + 1):
            dx = (X_pred[i] - self.x.flatten()).reshape(-1, 1)
            self.P += self.Wc[i] * (dx @ dx.T)

        self.sigma_pts_pred = X_pred

    def update(self, y):
        # Transform predicted sigma points through measurement function h(x)
        Y = []
        for x in self.sigma_pts_pred:
            x_input = np.dot(self.T_real, x[:self.nx].reshape(-1, 1)).T
            y_pred = self.get_y(x_input.T) + x[self.nx:].reshape(1, -1)  # Add disturbance part
            Y.append(y_pred.flatten())
        Y = np.array(Y)
        y_mean = np.sum(self.Wm[:, None] * Y, axis=0).reshape(-1, 1)

        # Innovation covariance
        Pyy = self.R.copy()
        for i in range(2 * self.n + 1):
            dy = (Y[i] - y_mean.flatten()).reshape(-1, 1)
            Pyy += self.Wc[i] * dy @ dy.T
        
        # Cross-covariance
        Pxy = np.zeros((self.n, y_mean.shape[0]))
        for i in range(2 * self.n + 1):
            dx = (self.sigma_pts_pred[i] - self.x.flatten()).reshape(-1, 1)
            dy = (Y[i] - y_mean.flatten()).reshape(-1, 1)
            Pxy += self.Wc[i] * dx @ dy.T

        # Kalman gain and update
        K = Pxy @ np.linalg.inv(Pyy)
        self.x = self.x + K @ (y.reshape(-1, 1) - y_mean)
        self.P = self.P - K @ Pyy @ K.T

    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.x

    
class EEKF():
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
    
    def get_x(self, y):
        x = self.problem.nodes[0]({"Y0": torch.from_numpy(y.T).float()})
        return x["x"].detach().numpy().reshape(1,-1)
    
    def predict(self, u):
        self.x = self.A @ self.x.reshape(-1,1) + self.B @ u.reshape(-1,1)
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x, self.P
    
    def update(self, y):
        y_pred = self.get_y(self.T_real@self.x.T[0,0:self.nx]) + self.x.T[0,self.nx:]
        J = evaluate_jacobian(self.problem.nodes[4], torch.tensor(self.T_real@self.x[:self.nx]).T[0])
        self.H = np.hstack([J@self.T_real, np.eye(self.nd)])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - y_pred).T
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        
        # Update the state with the estimated state from the output
        x_trans = self.get_x(y).T
        print(x_trans)
        print(self.x)
        x_diff = np.zeros((self.x.shape[0],1))
        x_diff[:self.nx] = np.linalg.inv(self.T_real)@x_trans - self.x[:self.nx].reshape(-1,1)
        Kx = self.P @ np.linalg.inv(self.P @ np.eye(self.x.shape[0]))
        self.x = self.x + Kx @ x_diff
        print(self.x)
        self.P = (np.eye(self.P.shape[0]) - Kx) @ self.P
        
        
    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.x
    
class EKF_C():
    def __init__(self, A, B, C, x0, P0, problem, Q, R, disturbance=0, T_real = None):
        self.disturbance = disturbance
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.C = C
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
        J = self.C
        self.H = np.hstack([J@self.T_real, np.eye(self.nd)])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - y_pred).T
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        
    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.x
    
class TargetEstimation():
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        self.nz = A.shape[0]
        self.ny = C.shape[0]
        self.nu = B.shape[1]
        loaded_setup = joblib.load("sim_setup.pkl")
        self.Qy = loaded_setup["Qy"]
        self.u_min = loaded_setup["u_min"]
        self.u_max = loaded_setup["u_max"]
        self.y_min = loaded_setup["y_min"]
        self.y_max = loaded_setup["y_max"]
        
        self.build_problem()

    def build_problem(self):
        self.z_s = cp.Variable(self.nz)
        self.y_s = cp.Variable(self.ny)
        self.u_s = cp.Variable(self.nu)
        self.d0 = cp.Parameter(self.ny)
        self.y_sp = cp.Parameter(self.ny)
        
        constraints_s = [self.z_s == self.A @ self.z_s + self.B @ self.u_s]
        constraints_s += [self.y_s == self.C @ self.z_s + self.d0]
        constraints_s += [self.u_min <= self.u_s, self.u_s <= self.u_max]
        constraints_s += [self.y_min <= self.y_s, self.y_s <= self.y_max]

        cost_s = 0
        cost_s += cp.quad_form(self.y_s - self.y_sp, self.Qy)

        self.te = cp.Problem(cp.Minimize(cost_s), constraints_s)
        
    def get_target(self, d0, y_sp):
        self.d0.value = d0.flatten()
        self.y_sp.value = y_sp.flatten()
        
        # solve the problem
        self.te.solve(solver=cp.GUROBI,TimeLimit=60,BarIterLimit=1e6)
        
        if self.te.status != cp.OPTIMAL:
            print("Target estimation problem is not optimal")
            print(self.te.status)
            print(self.te.solver_stats.solve_time)
            print(self.te.solver_stats.num_iters)
            raise RuntimeError("Solver did not return an optimal solution")
            
        return self.z_s.value, self.y_s.value
    
        
    
class MPC():
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        self.nz = A.shape[0]
        self.ny = C.shape[0]
        self.nu = B.shape[1]
        loaded_setup = joblib.load("sim_setup.pkl")
        Qy = loaded_setup["Qy"]
        Qz = C.T@Qy@C + 1e-8 * np.eye(A.shape[0])
        self.Qu = loaded_setup["Qu"]
        self.N = loaded_setup["N"]
        self.u_min = loaded_setup["u_min"]
        self.u_max = loaded_setup["u_max"]
        self.y_min = loaded_setup["y_min"]
        self.y_max = loaded_setup["y_max"]
        
        self.build_problem(Qz)
    
    def build_problem(self, Qz):
        '''
        Build the MPC problem using cvxpy
        '''
        # parameters
        self.z0 = cp.Parameter(self.nz)
        self.d0 = cp.Parameter(self.ny)
        self.u_prev = cp.Parameter(self.nu)
        self.z_ref = cp.Parameter(self.nz)

        # optimized variables
        z = cp.Variable((self.nz, self.N + 1))
        self.u = cp.Variable((self.nu, self.N)) 
        
        # building the problem
        constraints = [z[:, 0] == self.z0]
        cost = 0

        for k in range(self.N):
            constraints += [
                z[:, k+1] == self.A @ z[:, k] + self.B @ self.u[:,k],
                self.u_min <= self.u[:, k], self.u[:, k] <= self.u_max,
                self.y_min <= self.C @ z[:, k] + self.d0, self.C @ z[:, k] + self.d0 <= self.y_max
            ]
            if k == 0:
                cost += cp.quad_form(z[:, k] - self.z_ref, Qz) + cp.quad_form(self.u[:, 0] - self.u_prev, self.Qu)
            else:
                cost += cp.quad_form(z[:, k] - self.z_ref, Qz) + cp.quad_form(self.u[:, k] - self.u[:, k-1], self.Qu)
                
        self.mpc = cp.Problem(cp.Minimize(cost), constraints)
        
    def get_u_optimal(self, z0, d0, u_prev, z_ref):
        '''
        Get the optimal control input solving the MPC problem
        '''
        self.z0.value = z0.flatten()
        self.d0.value = d0.flatten()
        self.u_prev.value = u_prev.flatten()
        self.z_ref.value = z_ref.flatten()
        # solve the problem
        self.mpc.solve(solver=cp.GUROBI,TimeLimit=60,BarIterLimit=1e6)#, BarConvTol=1e-6)
        
        if self.mpc.status != cp.OPTIMAL:
            print("MPC problem is not optimal")
            print(self.mpc.status)
            print(self.mpc.solver_stats.solve_time)
            print(self.mpc.solver_stats.num_iters)
            raise RuntimeError("Solver did not return an optimal solution")
             
        return self.u[:, 0].value
    
class TaylorTargetEstimation():
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.nz = A.shape[0]
        self.nu = B.shape[1]
        loaded_setup = joblib.load("sim_setup.pkl")
        self.Qy = loaded_setup["Qy"]
        self.u_min = loaded_setup["u_min"]
        self.u_max = loaded_setup["u_max"]
        self.y_min = loaded_setup["y_min"]
        self.y_max = loaded_setup["y_max"]
        self.ny = self.y_max.shape[0]
        
        self.build_problem()

    def build_problem(self):
        self.z_s = cp.Variable(self.nz)
        self.y_s = cp.Variable(self.ny)
        self.u_s = cp.Variable(self.nu)
        self.d0 = cp.Parameter(self.ny)
        self.y_sp = cp.Parameter(self.ny)
        self.y_k = cp.Parameter(self.ny)
        self.z_k = cp.Parameter(self.nz)
        self.C_k = cp.Parameter((self.ny, self.nz))
        
        constraints_s = [self.z_s == self.A @ self.z_s + self.B @ self.u_s]
        constraints_s += [self.y_s == self.C_k @ self.z_s + self.y_k - self.C_k @ self.z_k + self.d0]
        constraints_s += [self.u_min <= self.u_s, self.u_s <= self.u_max]
        constraints_s += [self.y_min <= self.y_s, self.y_s <= self.y_max]

        
        cost_s = cp.quad_form(self.y_s - self.y_sp, self.Qy)

        self.te = cp.Problem(cp.Minimize(cost_s), constraints_s)
        
    def get_target(self, d0, y_sp, y_k, z_k, C_k):
        self.d0.value = d0.flatten()
        self.y_sp.value = y_sp.flatten()
        self.y_k.value = y_k.flatten()
        self.z_k.value = z_k.flatten()
        self.C_k.value = C_k
        # solve the problem
        self.te.solve(solver=cp.GUROBI)#,TimeLimit=60,BarIterLimit=1e6)
        
        if self.te.status != cp.OPTIMAL:
            print("Target estimation problem is not optimal")
            print(self.te.status)
            print(self.te.solver_stats.solve_time)
            print(self.te.solver_stats.num_iters)
            raise RuntimeError("Solver did not return an optimal solution")
            
        return self.z_s.value, self.y_s.value
    
class TaylorMPC():
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.nz = A.shape[0]
        self.nu = B.shape[1]
        loaded_setup = joblib.load("sim_setup.pkl")
        self.Qu = loaded_setup["Qu"]
        self.N = loaded_setup["N"]
        self.u_min = loaded_setup["u_min"]
        self.u_max = loaded_setup["u_max"]
        self.y_min = loaded_setup["y_min"]
        self.y_max = loaded_setup["y_max"]
        self.ny = self.y_max.shape[0]
        
    def build_problem(self, Qz):
        '''
        Build the MPC problem using cvxpy
        '''
        # parameters
        self.z0 = cp.Parameter(self.nz)
        self.d0 = cp.Parameter(self.ny)
        self.u_prev = cp.Parameter(self.nu)
        self.z_ref = cp.Parameter(self.nz)
        self.y_k = cp.Parameter(self.ny)
        self.z_k = cp.Parameter(self.nz)
        self.C_k = cp.Parameter((self.ny, self.nz))
        self.Qz = Qz

        # optimized variables
        z = cp.Variable((self.nz, self.N + 1))
        self.u = cp.Variable((self.nu, self.N)) 
        
        # building the problem
        constraints = [z[:, 0] == self.z0]
        cost = 0

        for k in range(self.N):
            constraints += [
                z[:, k+1] == self.A @ z[:, k] + self.B @ self.u[:,k],
                self.u_min <= self.u[:, k], self.u[:, k] <= self.u_max,
                self.y_min <= self.C_k @ z[:, k] + self.y_k - self.C_k @ self.z_k + self.d0, self.C_k @ z[:, k] + self.y_k - self.C_k @ self.z_k + self.d0 <= self.y_max
            ]
            if k == 0:
                cost += cp.quad_form(z[:, k] - self.z_ref, self.Qz) + cp.quad_form(self.u[:, 0] - self.u_prev, self.Qu)
            else:
                cost += cp.quad_form(z[:, k] - self.z_ref, self.Qz) + cp.quad_form(self.u[:, k] - self.u[:, k-1], self.Qu)
                
        self.mpc = cp.Problem(cp.Minimize(cost), constraints)
        
    def get_u_optimal(self, z0, d0, u_prev, z_ref, y_k, z_k, C_k):
        '''
        Get the optimal control input solving the MPC problem
        '''
        self.z0.value = z0.flatten()
        self.d0.value = d0.flatten()
        self.u_prev.value = u_prev.flatten()
        self.z_ref.value = z_ref.flatten()
        self.y_k.value = y_k.flatten()
        self.z_k.value = z_k.flatten()
        self.C_k.value = C_k
        # solve the problem
        self.mpc.solve(solver=cp.GUROBI,TimeLimit=60,BarIterLimit=1e6)#, BarConvTol=1e-6)
        
        if self.mpc.status != cp.OPTIMAL:
            print("MPC problem is not optimal")
            print(self.mpc.status)
            print(self.mpc.solver_stats.solve_time)
            print(self.mpc.solver_stats.num_iters)
            raise RuntimeError("Solver did not return an optimal solution")
             
        return self.u[:, 0].value