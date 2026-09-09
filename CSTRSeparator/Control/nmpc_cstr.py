"""Exact-model NMPC + discrete EKF for the CSTR–separator.

The ODE matches ``models.CSTRSeparator``. Both the EKF and the NMPC use the
same discrete RK4 map (several substeps per sample) so the observer is not
biased relative to the plant ``solve_ivp`` / RK45 stepper.

Feed temperature and composition (T10, T20, xA10, …) are parameters and must
be updated when the plant is disturbed.

A nonlinear target selector computes a feasible steady pair ``(y_s, u_s)``
from the current ``y_sp`` / ``u_sp`` (``Qy_te``, ``Qu_te``). NMPC tracks that
pair over the horizon with no preview, matching the N4SID loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import casadi as ca
import do_mpc
import numpy as np


X_NAMES = ["xA1", "xB1", "T1", "xA2", "xB2", "T2", "xA3", "xB3", "T3"]
U_NAMES = ["Q1", "Q2", "Q3", "F10", "F20", "Fr"]
Y_NAMES = ["T1", "T2", "T3", "xB3"]
P_NAMES = ["T10", "T20", "xA10", "xB10", "xA20", "xB20"]
NX, NU, NY, NP = 9, 6, 4, 6
N_RK4_SUB = 10


def _plant_constants():
    # Defaults of models.CSTRSeparator.__init__
    return {
        "rho": 0.15,
        "E1": 813.4,
        "E2": 1247.1,
        "T10": 313.0,
        "T20": 313.0,
        "V1": 89.4,
        "V2": 90.0,
        "V3": 13.27,
        "k1": 0.336,
        "k2": 0.089,
        "R": 8.314,
        "xA10": 1.0,
        "xB10": 0.0,
        "xA20": 1.0,
        "xB20": 0.0,
        "DH1": -40.0,
        "DH2": -50.0,
        "Cp": 2.5,
        "alpha_A": 3.5,
        "alpha_B": 0.5,
        "alpha_C": 1.1,
    }


class FeedParams:
    """Nominal Li & Swartz feed conditions; mutate when the plant is disturbed."""

    def __init__(self):
        c = _plant_constants()
        self.values = {name: float(c[name]) for name in P_NAMES}

    def set(self, name: str, value: float):
        if name not in self.values:
            raise AttributeError(f"Unknown feed parameter {name!r}")
        self.values[name] = float(value)

    def vector(self) -> np.ndarray:
        return np.array([self.values[n] for n in P_NAMES], dtype=float)


@dataclass
class CurrentRef:
    """N4SID-style setpoint: current y_ref / u_sp, repeated over the horizon."""

    y_ref: np.ndarray = field(default_factory=lambda: np.zeros(NY))
    u_ref: np.ndarray = field(default_factory=lambda: np.zeros(NU))

    def set(self, y_ref, u_ref):
        self.y_ref = np.asarray(y_ref, dtype=float).reshape(NY)
        self.u_ref = np.asarray(u_ref, dtype=float).reshape(NU)


def _merge_params(p_override=None):
    p = _plant_constants()
    if p_override is not None:
        p.update(p_override)
    return p


def cstr_rhs(x, u, p=None):
    """CasADi RHS. ``p`` may mix floats and SX (feed conditions)."""
    p = _merge_params(p)
    xA1, xB1, T1, xA2, xB2, T2, xA3, xB3, T3 = x
    Q1, Q2, Q3, F10, F20, Fr = u

    xA1 = ca.fmax(xA1, 0.0)
    xB1 = ca.fmax(xB1, 0.0)
    xA2 = ca.fmax(xA2, 0.0)
    xB2 = ca.fmax(xB2, 0.0)
    xA3 = ca.fmax(xA3, 0.0)
    xB3 = ca.fmax(xB3, 0.0)
    T1 = ca.fmax(T1, 1.0)
    T2 = ca.fmax(T2, 1.0)
    T3 = ca.fmax(T3, 1.0)

    xC3 = ca.fmax(1.0 - xA3 - xB3, 0.0)
    den = p["alpha_A"] * xA3 + p["alpha_B"] * xB3 + p["alpha_C"] * xC3
    den = ca.fmax(den, 1e-10)
    xAr = p["alpha_A"] * xA3 / den
    xBr = p["alpha_B"] * xB3 / den

    F1 = F10 + Fr
    F2 = F1 + F20
    Fp = F10 + F20
    Fr_plus_Fp = Fr + Fp

    r1_1 = p["k1"] * ca.exp(-p["E1"] / (p["R"] * T1)) * xA1
    r2_1 = p["k2"] * ca.exp(-p["E2"] / (p["R"] * T1)) * xB1
    r1_2 = p["k1"] * ca.exp(-p["E1"] / (p["R"] * T2)) * xA2
    r2_2 = p["k2"] * ca.exp(-p["E2"] / (p["R"] * T2)) * xB2

    dxA1 = (F10 / p["V1"]) * (p["xA10"] - xA1) + (Fr / p["V1"]) * (xAr - xA1) - r1_1
    dxB1 = (F10 / p["V1"]) * (p["xB10"] - xB1) + (Fr / p["V1"]) * (xBr - xB1) + r1_1 - r2_1
    dT1 = (
        (F10 / p["V1"]) * (p["T10"] - T1)
        + (Fr / p["V1"]) * (T3 - T1)
        + (-p["DH1"] / p["Cp"]) * r1_1
        + (-p["DH2"] / p["Cp"]) * r2_1
        + Q1 / (p["rho"] * p["Cp"] * p["V1"])
    )
    dxA2 = (F1 / p["V2"]) * (xA1 - xA2) + (F20 / p["V2"]) * (p["xA20"] - xA2) - r1_2
    dxB2 = (F1 / p["V2"]) * (xB1 - xB2) + (F20 / p["V2"]) * (p["xB20"] - xB2) + r1_2 - r2_2
    dT2 = (
        (F1 / p["V2"]) * (T1 - T2)
        + (F20 / p["V2"]) * (p["T20"] - T2)
        + (-p["DH1"] / p["Cp"]) * r1_2
        + (-p["DH2"] / p["Cp"]) * r2_2
        + Q2 / (p["rho"] * p["Cp"] * p["V2"])
    )
    dxA3 = (F2 / p["V3"]) * (xA2 - xA3) - (Fr_plus_Fp / p["V3"]) * (xAr - xA3)
    dxB3 = (F2 / p["V3"]) * (xB2 - xB3) - (Fr_plus_Fp / p["V3"]) * (xBr - xB3)
    dT3 = (F2 / p["V3"]) * (T2 - T3) + Q3 / (p["rho"] * p["Cp"] * p["V3"])
    return [dxA1, dxB1, dT1, dxA2, dxB2, dT2, dxA3, dxB3, dT3]


def measure_x(x):
    """y = [T1, T2, T3, xB3] from the 9-vector."""
    return ca.vertcat(x[2], x[5], x[8], x[7])


def rk4_map(x, u, p_map, Ts: float, n_sub: int = N_RK4_SUB):
    """Discrete state map: ``n_sub`` RK4 steps over one sample ``Ts``."""
    h = float(Ts) / int(n_sub)
    xk = x if isinstance(x, ca.SX) else ca.vertcat(*x)

    def rhs_at(xx):
        return ca.vertcat(*cstr_rhs([xx[i] for i in range(NX)], u, p_map))

    for _ in range(int(n_sub)):
        k1 = rhs_at(xk)
        k2 = rhs_at(xk + 0.5 * h * k1)
        k3 = rhs_at(xk + 0.5 * h * k2)
        k4 = rhs_at(xk + h * k3)
        xk = xk + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return xk


def build_model(Ts: float = 1.0, n_sub: int = N_RK4_SUB) -> do_mpc.model.Model:
    model = do_mpc.model.Model("discrete")
    xs: List = [model.set_variable("_x", name) for name in X_NAMES]
    us: List = [model.set_variable("_u", name) for name in U_NAMES]
    for name in Y_NAMES:
        model.set_variable("_tvp", f"{name}_ref")
    for name in U_NAMES:
        model.set_variable("_tvp", f"{name}_ref")
    ps = {name: model.set_variable("_p", name) for name in P_NAMES}

    x_next = rk4_map(ca.vertcat(*xs), us, ps, Ts, n_sub)
    for i, name in enumerate(X_NAMES):
        model.set_rhs(name, x_next[i])
    model.setup()
    return model


def _unscale_weights(Q_scaled: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale = np.asarray(scale, dtype=float).reshape(-1)
    return Q_scaled / (scale[:, None] * scale[None, :])


def build_mpc(
    model: do_mpc.model.Model,
    loaded_setup: Dict,
    scaler,
    scalerU,
    feed: FeedParams,
    current_ref: CurrentRef,
) -> do_mpc.controller.MPC:
    mpc = do_mpc.controller.MPC(model)
    Ts = float(loaded_setup.get("Ts", 1.0))
    mpc.set_param(
        n_horizon=int(loaded_setup["N"]),
        t_step=Ts,
        store_full_solution=True,
        nlpsol_opts={
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 400,
        },
    )

    Qy = _unscale_weights(np.asarray(loaded_setup["Qy"], dtype=float), scaler.scale_)
    Qu = _unscale_weights(np.asarray(loaded_setup["Qu"], dtype=float), scalerU.scale_)
    Qdu = _unscale_weights(np.asarray(loaded_setup["Qdu"], dtype=float), scalerU.scale_)

    y_vars = [model.x["T1"], model.x["T2"], model.x["T3"], model.x["xB3"]]
    y_ref = [model.tvp[f"{n}_ref"] for n in Y_NAMES]
    u_vars = [model.u[n] for n in U_NAMES]
    u_ref = [model.tvp[f"{n}_ref"] for n in U_NAMES]
    lterm = 0
    for i in range(NY):
        lterm = lterm + Qy[i, i] * (y_vars[i] - y_ref[i]) ** 2
    for i in range(NU):
        lterm = lterm + Qu[i, i] * (u_vars[i] - u_ref[i]) ** 2
    mpc.set_objective(mterm=lterm, lterm=lterm)
    mpc.set_rterm(**{name: float(Qdu[i, i]) for i, name in enumerate(U_NAMES)})

    u_min = np.asarray(loaded_setup["u_min_ns"], dtype=float).reshape(-1)
    u_max = np.asarray(loaded_setup["u_max_ns"], dtype=float).reshape(-1)
    for i, name in enumerate(U_NAMES):
        mpc.bounds["lower", "_u", name] = float(u_min[i])
        mpc.bounds["upper", "_u", name] = float(u_max[i])

    y_min = np.asarray(loaded_setup["y_min_ns"], dtype=float).reshape(-1)
    y_max = np.asarray(loaded_setup["y_max_ns"], dtype=float).reshape(-1)
    for i, name in enumerate(Y_NAMES):
        mpc.bounds["lower", "_x", name] = float(y_min[i])
        mpc.bounds["upper", "_x", name] = float(y_max[i])

    n_horizon = int(loaded_setup["N"])
    tvp_template = mpc.get_tvp_template()
    p_template = mpc.get_p_template(1)

    def tvp_fun(_t_now):
        for i in range(n_horizon + 1):
            for j, name in enumerate(Y_NAMES):
                tvp_template["_tvp", i, f"{name}_ref"] = float(current_ref.y_ref[j])
            for j, name in enumerate(U_NAMES):
                tvp_template["_tvp", i, f"{name}_ref"] = float(current_ref.u_ref[j])
        return tvp_template

    def p_fun(_t_now):
        p_template["_p", 0] = feed.vector()
        return p_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.set_p_fun(p_fun)
    mpc.setup()
    return mpc


class TargetSelector:
    """Steady-state target, same role as N4SID ``TargetEstimation``.

    ``min  ||y - y_sp||_{Qy_te}^2 + ||u - u_sp||_{Qu_te}^2``
    s.t. discrete SS ``x = F(x,u,p)``, input/output bounds.

    NMPC then tracks the feasible pair ``(y_s, u_s)``, so ``Qu > 0`` cannot
    ask the plant to sit on an unreachable ``(y_sp, u_sp)``. Trade-off between
    outputs and ``u_sp`` is set by ``Qy_te`` / ``Qu_te`` in ``setup.py``.
    """

    def __init__(
        self,
        loaded_setup: Dict,
        scaler,
        scalerU,
        feed: FeedParams,
        Ts: float = 1.0,
        n_sub: int = N_RK4_SUB,
    ):
        self.feed = feed
        self.Ts = float(Ts)
        Qy = _unscale_weights(np.asarray(loaded_setup["Qy_te"], dtype=float), scaler.scale_)
        Qu = _unscale_weights(np.asarray(loaded_setup["Qu_te"], dtype=float), scalerU.scale_)
        self.u_min = np.asarray(loaded_setup["u_min_ns"], dtype=float).reshape(NU)
        self.u_max = np.asarray(loaded_setup["u_max_ns"], dtype=float).reshape(NU)
        self.y_min = np.asarray(loaded_setup["y_min_ns"], dtype=float).reshape(NY)
        self.y_max = np.asarray(loaded_setup["y_max_ns"], dtype=float).reshape(NY)

        x = ca.SX.sym("x", NX)
        u = ca.SX.sym("u", NU)
        p = ca.SX.sym("p", NP)
        y_sp = ca.SX.sym("y_sp", NY)
        u_sp = ca.SX.sym("u_sp", NU)
        p_map = {name: p[i] for i, name in enumerate(P_NAMES)}
        u_list = [u[i] for i in range(NU)]
        x_next = rk4_map(x, u_list, p_map, self.Ts, n_sub)
        y = measure_x(x)
        dy = y - y_sp
        du = u - u_sp
        cost = dy.T @ ca.DM(Qy) @ dy + du.T @ ca.DM(Qu) @ du

        nlp = {
            "x": ca.vertcat(x, u),
            "p": ca.vertcat(p, y_sp, u_sp),
            "f": cost,
            "g": ca.vertcat(x_next - x, y),
        }
        self._solver = ca.nlpsol(
            "target",
            "ipopt",
            nlp,
            {
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "print_time": 0,
                "ipopt.max_iter": 200,
            },
        )
        x_lb = np.array([0.0, 0.0, self.y_min[0], 0.0, 0.0, self.y_min[1], 0.0, 0.0, self.y_min[2]])
        x_ub = np.array([1.0, 1.0, self.y_max[0], 1.0, 1.0, self.y_max[1], 1.0, 1.0, self.y_max[2]])
        self._lbx = np.concatenate([x_lb, self.u_min])
        self._ubx = np.concatenate([x_ub, self.u_max])
        self._lbg = np.concatenate([np.zeros(NX), self.y_min])
        self._ubg = np.concatenate([np.zeros(NX), self.y_max])
        self.x_s = None
        self.u_s = None

    def get_target(self, y_sp, u_sp, x_guess=None, u_guess=None):
        y_sp = np.asarray(y_sp, dtype=float).reshape(NY)
        u_sp = np.asarray(u_sp, dtype=float).reshape(NU)
        if x_guess is None:
            x_guess = self.x_s if self.x_s is not None else np.array(
                [0.25, 0.42, y_sp[0], 0.10, 0.41, y_sp[1], 0.02, y_sp[3], y_sp[2]]
            )
        if u_guess is None:
            u_guess = self.u_s if self.u_s is not None else u_sp
        w0 = np.concatenate(
            [np.asarray(x_guess, dtype=float).reshape(NX), np.asarray(u_guess, dtype=float).reshape(NU)]
        )
        p = np.concatenate([self.feed.vector(), y_sp, u_sp])
        sol = self._solver(
            x0=w0, p=p, lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg
        )
        stats = self._solver.stats()
        if not stats.get("success", False):
            print("Target NLP:", stats.get("return_status"))
            raise RuntimeError("Target selector did not converge")
        w = np.asarray(sol["x"]).reshape(-1)
        self.x_s = w[:NX]
        self.u_s = w[NX:]
        y_s = np.array([self.x_s[2], self.x_s[5], self.x_s[8], self.x_s[7]])
        return self.x_s.copy(), y_s, self.u_s.copy()


class DiscreteEKF:
    """RK4 EKF on the same discrete map as NMPC. Measurements: [T1, T2, T3, xB3]."""

    def __init__(self, x0, P0, Q, R, feed: FeedParams, Ts=1.0, n_sub: int = N_RK4_SUB):
        self.Ts = float(Ts)
        self.feed = feed
        x = ca.SX.sym("x", NX)
        u = ca.SX.sym("u", NU)
        p = ca.SX.sym("p", NP)
        p_map = {name: p[i] for i, name in enumerate(P_NAMES)}
        u_list = [u[i] for i in range(NU)]
        x_next = rk4_map(x, u_list, p_map, self.Ts, n_sub)
        y = measure_x(x)

        self._F = ca.Function("F", [x, u, p], [x_next])
        self._A = ca.Function("A", [x, u, p], [ca.jacobian(x_next, x)])
        self._h = ca.Function("h", [x], [y])
        self._H = ca.Function("H", [x], [ca.jacobian(y, x)])

        self.x = np.asarray(x0, dtype=float).reshape(NX)
        self.P = np.asarray(P0, dtype=float).reshape(NX, NX)
        self.Q = np.asarray(Q, dtype=float).reshape(NX, NX)
        self.R = np.asarray(R, dtype=float).reshape(NY, NY)

    def step(self, u, y):
        u = np.asarray(u, dtype=float).reshape(NU)
        y = np.asarray(y, dtype=float).reshape(NY)
        p = self.feed.vector()
        x_pred = np.asarray(self._F(self.x, u, p)).reshape(NX)
        A = np.asarray(self._A(self.x, u, p)).reshape(NX, NX)
        P_pred = A @ self.P @ A.T + self.Q
        H = np.asarray(self._H(x_pred)).reshape(NY, NX)
        y_pred = np.asarray(self._h(x_pred)).reshape(NY)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x = (x_pred + K @ (y - y_pred)).reshape(NX)
        self.P = (np.eye(NX) - K @ H) @ P_pred
        return self.x.copy()


def default_ekf_covariances(loaded_setup: Dict):
    """Physical-unit EKF covariances. setup Q/R belong to the scaled N4SID KF."""
    noise = np.asarray(loaded_setup.get("noise_sigma", np.zeros(NY)), dtype=float).reshape(NY)
    r_diag = np.maximum(noise**2, np.array([0.05, 0.05, 0.05, 1e-6]) ** 2)
    q_diag = np.array([1e-6, 1e-6, 1e-2, 1e-6, 1e-6, 1e-2, 1e-6, 1e-6, 1e-2])
    P0 = np.diag(10.0 * q_diag)
    return P0, np.diag(q_diag), np.diag(r_diag)
