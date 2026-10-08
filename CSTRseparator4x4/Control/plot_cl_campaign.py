#!/usr/bin/env python3
"""Paper-style figures: closed-loop grade campaign, original weights.

Square 4x4 CSTR-separator variant (inputs [Q1, Q2, Q3, F10]; F20, Fr fixed).
Outputs and inputs are separate figures. Trajectories: NMPC, N4SID, CT
(linear decoder), T3D3 (purple dashed), T2D2 on top (red dash-dotted).

Figures are written to ``document/figures``, which ``document/main.tex`` includes.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, MaxNLocator, MultipleLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP = Path(__file__).resolve().parent / "results" / "weight_sweep"
# Closed-loop N4SID in the weight sweep chatters on T3; the notebook run does not.
N4SID_DIR = (
    Path(__file__).resolve().parent / "results" / "parsimk_notebook_qd" / "N4SID_original"
)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = REPO_ROOT / "document" / "figures"

# elsarticle preprint, 12 pt, one column: 384 pt textwidth, figures at 0.95\linewidth
BODY_PT = 12
TICK_PT = 10
TEXT_IN = 384.0 / 72.0
INCLUDE_FRAC = 0.95
FIG_W_IN = INCLUDE_FRAC * TEXT_IN
FIG_H_Y_IN = 6.2
FIG_H_U_IN = 5.0
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
# Output figure only. Campaign window 250–1250 s; the axis origin is 250 s.
T_LO = 250.0
T_HI = 1250.0
T_ORIGIN = 250.0
ZOOM_LO = 990.0
ZOOM_HI = 1070.0

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
    if controller == "N4SID":
        path = N4SID_DIR / "trajectories.npz"
    else:
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
    """Campaign from 250 s to 1250 s, with a 990–1070 s close-up.

    The nonlinear MPC trajectory is omitted. The setpoint stays. Both columns share
    the vertical scale and count time from 250 s, so that instant is 0.
    """
    y_series, _, _, _, _ = _load_cl_series()
    y_series = tuple(
        s for s in y_series if not (s[2] == COLOR_NMPC and s[3] == "-")
    )
    t_lo, t_hi = T_LO - T_ORIGIN, T_HI - T_ORIGIN
    z_lo, z_hi = ZOOM_LO - T_ORIGIN, ZOOM_HI - T_ORIGIN

    # Shorter than the single-column stack so the two-column figure and its caption fit the page.
    fig = plt.figure(figsize=(FIG_W_IN, 5.32))
    gs = fig.add_gridspec(
        4, 2, width_ratios=(1.85, 1.0), wspace=0.28, hspace=0.18,
    )
    left_axes = []
    right_axes = []
    for i in range(4):
        ax_l = fig.add_subplot(gs[i, 0])
        ax_r = fig.add_subplot(gs[i, 1], sharey=ax_l)
        for ax, x0, x1 in ((ax_l, t_lo, t_hi), (ax_r, z_lo, z_hi)):
            for t_ev in EVENTS:
                ax.axvline(t_ev - T_ORIGIN, color=COLOR_EVENT, linewidth=LW_EVENT, zorder=1)
            for y, t, color, ls, lw, z in y_series:
                ax.plot(t - T_ORIGIN, y[i], color=color, linestyle=ls, linewidth=lw, zorder=z)
            ax.set_xlim(x0, x1)
            ax.grid(True)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
        ax_l.set_ylabel(YLABELS[i], fontsize=BODY_PT)
        ax_r.tick_params(axis="y", which="both", left=False, labelleft=False)
        left_axes.append(ax_l)
        right_axes.append(ax_r)

    for ax in (*left_axes[:-1], *right_axes[:-1]):
        ax.tick_params(axis="x", which="both", labelbottom=False)
    left_axes[-1].xaxis.set_major_locator(FixedLocator([0, 250, 500, 750, 1000]))
    right_axes[-1].xaxis.set_major_locator(FixedLocator([740, 780, 820]))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.99, bottom=0.10)
    x_mid = 0.5 * (
        left_axes[-1].get_position().x1 + right_axes[-1].get_position().x0
    )
    fig.text(
        x_mid, 0.012, r"Time $t$ [s]", ha="center", va="bottom", fontsize=BODY_PT,
    )
    out = OUT_DIR / "cl_trajectories.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def plot_inputs() -> Path:
    """Same window, close-up, and time origin as the output figure.

    The nonlinear MPC trajectory is omitted. The right column shares the vertical
    scale, including the input-bound bands, and carries no separate y-axis.
    """
    _, u_series, t_u, u_min, u_max = _load_cl_series()
    u_series = tuple(
        s for s in u_series if not (s[1] == COLOR_NMPC and s[2] == "-")
    )
    t_lo, t_hi = T_LO - T_ORIGIN, T_HI - T_ORIGIN
    z_lo, z_hi = ZOOM_LO - T_ORIGIN, ZOOM_HI - T_ORIGIN
    t_plot = t_u - T_ORIGIN

    fig = plt.figure(figsize=(FIG_W_IN, 4.85))
    gs = fig.add_gridspec(
        4, 2, width_ratios=(1.85, 1.0), wspace=0.28, hspace=0.18,
    )
    left_axes = []
    right_axes = []
    for j in range(4):
        ax_l = fig.add_subplot(gs[j, 0])
        ax_r = fig.add_subplot(gs[j, 1], sharey=ax_l)
        span = float(u_max[j] - u_min[j])
        pad = 0.08 * span
        ymin, ymax = u_min[j] - pad, u_max[j] + pad
        for ax, x0, x1 in ((ax_l, t_lo, t_hi), (ax_r, z_lo, z_hi)):
            for t_ev in EVENTS:
                ax.axvline(t_ev - T_ORIGIN, color=COLOR_EVENT, linewidth=LW_EVENT, zorder=1)
            ax.axhspan(ymin, u_min[j], color=COLOR_BOUND, alpha=0.10, zorder=0, lw=0)
            ax.axhspan(u_max[j], ymax, color=COLOR_BOUND, alpha=0.10, zorder=0, lw=0)
            for u, color, ls, lw, z in u_series:
                ax.plot(t_plot, u[j], color=color, linestyle=ls, linewidth=lw, zorder=z)
            ax.set_xlim(x0, x1)
            ax.grid(True)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
        ax_l.set_ylabel(ULABELS[j], fontsize=BODY_PT)
        ax_l.set_ylim(ymin, ymax)
        ax_r.tick_params(axis="y", which="both", left=False, labelleft=False)
        left_axes.append(ax_l)
        right_axes.append(ax_r)

    for ax in (*left_axes[:-1], *right_axes[:-1]):
        ax.tick_params(axis="x", which="both", labelbottom=False)
    left_axes[-1].xaxis.set_major_locator(FixedLocator([0, 250, 500, 750, 1000]))
    right_axes[-1].xaxis.set_major_locator(FixedLocator([740, 780, 820]))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.99, bottom=0.10)
    x_mid = 0.5 * (
        left_axes[-1].get_position().x1 + right_axes[-1].get_position().x0
    )
    fig.text(
        x_mid, 0.012, r"Time $t$ [s]", ha="center", va="bottom", fontsize=BODY_PT,
    )
    out = OUT_DIR / "cl_inputs.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


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


def plot_cost_bars() -> Path:
    """Grouped bars of J / J_{linear h} for three output-weight tunings.

    Costs are the tabulated totals in document/main.tex. Columns are the
    Koopman model with the linear decoder, the proposed linearization at
    the previous target, and the proposed linearization at the current
    estimate. Each row is divided by the linear-decoder cost of that tuning.
    """
    apply_style()
    # Rows: Qy/5, original Qy, 5 Qy.
    # Cols: linear h, previous target, current estimate.
    costs = np.array(
        [
            [30.0, 28.4, 28.3],
            [125.8, 111.5, 112.2],
            [605.9, 504.3, 520.6],
        ],
        dtype=float,
    )
    rel = 100.0 * costs / costs[:, [0]]
    colors = (COLOR_CT, COLOR_T2D2, COLOR_T3D3)
    labels = (
        r"$\frac{1}{5}Q_{\mathrm{y}}$",
        r"$Q_{\mathrm{y}}$",
        r"$5Q_{\mathrm{y}}$",
    )
    n_groups, n_series = rel.shape
    x = np.arange(n_groups, dtype=float)
    width = 0.22
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(FIG_W_IN, 3.15))
    for i, color in enumerate(colors):
        ax.bar(
            x + offsets[i],
            rel[:, i],
            width=width * 0.92,
            color=color,
            edgecolor="black",
            linewidth=0.6 if i >= 1 else 0.4,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_PT)
    ax.set_xlabel(r"Output weight", fontsize=BODY_PT)
    ax.set_ylabel(r"$J/J_{\mathrm{linear}\,h}$", fontsize=BODY_PT)
    ax.set_xlim(-0.55, n_groups - 0.45)
    ax.set_ylim(80, 100)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.97, bottom=0.18)
    out = OUT_DIR / "cl_cost_bars.pdf"
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
