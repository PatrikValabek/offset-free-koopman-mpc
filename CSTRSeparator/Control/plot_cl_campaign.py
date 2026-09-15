#!/usr/bin/env python3
"""Paper figures: closed-loop grade campaign, original weights.

Outputs and inputs are separate figures. Trajectories: NMPC, N4SID, CT
(linear decoder), T3D3 (purple dashed), T2D2 on top (red dash-dotted).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP = Path(__file__).resolve().parent / "results" / "weight_sweep"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = REPO_ROOT / "document" / "figures"

# elsarticle preprint, 12 pt, one column: 384 pt textwidth, figures at 0.95\linewidth
BODY_PT = 12
TICK_PT = 10
TEXT_IN = 384.0 / 72.0
INCLUDE_FRAC = 0.95
FIG_W_IN = INCLUDE_FRAC * TEXT_IN
FIG_H_Y_IN = 6.2
FIG_H_U_IN = 6.8
FIG_H_DIST_IN = 6.2

COLOR_NMPC = "#000000"
COLOR_N4SID = "#008000"  # parsimk
COLOR_CT = "#FFA500"  # koopmanc
COLOR_T2D2 = "#8B0000"  # deepkoopman
COLOR_T3D3 = "#7030A0"  # koopmanb, purple
COLOR_EVENT = "#808080"
COLOR_BOUND = "#000000"

LW_NMPC = 1.2
LW_BASE = 1.2
LW_T2D2 = 1.8
LW_T3D3 = 1.6
LW_REF = 1.0
LW_EVENT = 0.6

TS = 1.0
EVENTS = (350.0, 600.0, 850.0, 1000.0)

YLABELS = (
    r"$T_1$ [K]",
    r"$T_2$ [K]",
    r"$T_3$ [K]",
    r"$x_{\mathrm{B}3}$ [-]",
)
ULABELS = (
    r"$Q_1$ [kJ/s]",
    r"$Q_2$ [kJ/s]",
    r"$Q_3$ [kJ/s]",
    r"$F_{10}$ [m$^3$/s]",
    r"$F_{20}$ [m$^3$/s]",
    r"$F_{\mathrm{r}}$ [m$^3$/s]",
)
DLABELS = (
    r"$\hat{d}_{T_1}$ [K]",
    r"$\hat{d}_{T_2}$ [K]",
    r"$\hat{d}_{T_3}$ [K]",
    r"$\hat{d}_{x_{\mathrm{B}3}}$ [-]",
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.labelsize": BODY_PT,
            "xtick.labelsize": TICK_PT,
            "ytick.labelsize": TICK_PT,
            "legend.fontsize": TICK_PT,
            "axes.titlesize": BODY_PT,
            "axes.linewidth": 0.6,
            "lines.linewidth": LW_BASE,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def load_run(controller: str) -> np.lib.npyio.NpzFile:
    path = SWEEP / f"{controller}_original" / "trajectories.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path)


def event_lines(ax) -> None:
    for t in EVENTS:
        ax.axvline(t, color=COLOR_EVENT, linewidth=LW_EVENT, zorder=1)


def style_axis(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=BODY_PT)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
    ax.grid(True)
    ax.set_xlim(-20.0, 1500.0)


def _load_cl_series():
    nmpc = load_run("NMPC")
    n4sid = load_run("N4SID")
    ct = load_run("CT")
    t2 = load_run("T2D2")
    t3 = load_run("T3D3")
    t_y = np.arange(nmpc["y_true_ns"].shape[1]) * TS
    t_u = np.arange(nmpc["u_sim_ns"].shape[1]) * TS
    t_r = np.arange(nmpc["reference_ns"].shape[1]) * TS
    y_series = (
        (nmpc["reference_ns"], t_r, COLOR_NMPC, "--", LW_REF, 2),
        (nmpc["y_true_ns"], t_y, COLOR_NMPC, "-", LW_NMPC, 4),
        (n4sid["y_true_ns"], t_y, COLOR_N4SID, ":", LW_BASE, 3),
        (ct["y_true_ns"], t_y, COLOR_CT, "--", LW_BASE, 3),
        (t3["y_true_ns"], t_y, COLOR_T3D3, "--", LW_T3D3, 5),
        (t2["y_true_ns"], t_y, COLOR_T2D2, "-.", LW_T2D2, 6),
    )
    u_series = (
        (nmpc["u_sim_ns"], COLOR_NMPC, "-", LW_NMPC, 4),
        (n4sid["u_sim_ns"], COLOR_N4SID, ":", LW_BASE, 3),
        (ct["u_sim_ns"], COLOR_CT, "--", LW_BASE, 3),
        (t3["u_sim_ns"], COLOR_T3D3, "--", LW_T3D3, 5),
        (t2["u_sim_ns"], COLOR_T2D2, "-.", LW_T2D2, 6),
    )
    u_min = np.asarray(nmpc["u_min_ns"], dtype=float)
    u_max = np.asarray(nmpc["u_max_ns"], dtype=float)
    return y_series, u_series, t_u, u_min, u_max


def _finish_stack(fig, axes, out: Path, bottom: float) -> Path:
    axes[-1].set_xlabel(r"Time $t$ [s]", fontsize=BODY_PT)
    axes[-1].xaxis.set_major_locator(MultipleLocator(500))
    fig.subplots_adjust(left=0.20, right=0.98, top=0.99, bottom=bottom, hspace=0.18)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def plot_outputs() -> Path:
    y_series, _, _, _, _ = _load_cl_series()
    fig, axes = plt.subplots(
        4, 1, figsize=(FIG_W_IN, FIG_H_Y_IN), sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    for i, ax in enumerate(axes):
        event_lines(ax)
        for y, t, color, ls, lw, z in y_series:
            ax.plot(t, y[i], color=color, linestyle=ls, linewidth=lw, zorder=z)
        style_axis(ax, YLABELS[i])
    return _finish_stack(fig, axes, OUT_DIR / "cl_trajectories.pdf", 0.08)


def plot_inputs() -> Path:
    _, u_series, t_u, u_min, u_max = _load_cl_series()
    fig, axes = plt.subplots(
        6, 1, figsize=(FIG_W_IN, FIG_H_U_IN), sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    for j, ax in enumerate(axes):
        event_lines(ax)
        span = float(u_max[j] - u_min[j])
        pad = 0.08 * span
        ymin, ymax = u_min[j] - pad, u_max[j] + pad
        ax.axhspan(ymin, u_min[j], color=COLOR_BOUND, alpha=0.10, zorder=0, lw=0)
        ax.axhspan(u_max[j], ymax, color=COLOR_BOUND, alpha=0.10, zorder=0, lw=0)
        for u, color, ls, lw, z in u_series:
            ax.plot(t_u, u[j], color=color, linestyle=ls, linewidth=lw, zorder=z)
        style_axis(ax, ULABELS[j])
        ax.set_ylim(ymin, ymax)
    return _finish_stack(fig, axes, OUT_DIR / "cl_inputs.pdf", 0.055)


def d_physical(run, scale: np.ndarray) -> np.ndarray:
    return np.asarray(run["d_est"], dtype=float) * scale[:, None]


def plot_disturbances() -> Path:
    scaler = joblib.load(DATA_DIR / "scaler_cstr_separator.pkl")
    scale = np.asarray(scaler.scale_, dtype=float)
    n4sid = load_run("N4SID")
    ct = load_run("CT")
    t2 = load_run("T2D2")
    t3 = load_run("T3D3")
    t = np.arange(n4sid["d_est"].shape[1]) * TS
    series = (
        (d_physical(n4sid, scale), COLOR_N4SID, ":", LW_BASE, 3),
        (d_physical(ct, scale), COLOR_CT, "--", LW_BASE, 3),
        (d_physical(t3, scale), COLOR_T3D3, "--", LW_T3D3, 5),
        (d_physical(t2, scale), COLOR_T2D2, "-.", LW_T2D2, 6),
    )

    fig, axes = plt.subplots(
        4, 1, figsize=(FIG_W_IN, FIG_H_DIST_IN), sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    for i, ax in enumerate(axes):
        event_lines(ax)
        for d, color, ls, lw, z in series:
            ax.plot(t, d[i], color=color, linestyle=ls, linewidth=lw, zorder=z)
        style_axis(ax, DLABELS[i])

    axes[-1].set_xlabel(r"Time $t$ [s]", fontsize=BODY_PT)
    axes[-1].xaxis.set_major_locator(MultipleLocator(500))
    fig.subplots_adjust(left=0.20, right=0.98, top=0.99, bottom=0.08, hspace=0.18)
    out = OUT_DIR / "cl_disturbances.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def pdf_width_in(path: Path) -> float:
    raw = path.read_bytes()
    i = raw.find(b"/MediaBox")
    if i < 0:
        raise RuntimeError(f"no MediaBox in {path}")
    snippet = raw[i : i + 80].decode("latin1")
    nums = []
    token = ""
    for ch in snippet:
        if ch in "0123456789.+-":
            token += ch
        elif token:
            nums.append(float(token))
            token = ""
    if len(nums) < 4:
        raise RuntimeError(f"could not parse MediaBox of {path}: {snippet!r}")
    return (nums[2] - nums[0]) / 72.0


def main() -> None:
    apply_style()
    outs = (plot_outputs(), plot_inputs(), plot_disturbances())
    for p in outs:
        w = pdf_width_in(p)
        print(f"Wrote {p}  MediaBox width {w:.4f} in  (designed {FIG_W_IN:.4f} in)")


if __name__ == "__main__":
    main()
