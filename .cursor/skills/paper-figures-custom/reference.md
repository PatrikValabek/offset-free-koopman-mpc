# Paper-figure-custom reference

IEEE `ieeeconf` 10 pt two-column defaults for this paper. Distilled from `paper/ieeeconf.cls`, `paper/main.tex`, and **paper** cells in `code/plots.ipynb` (not the presentation cells).

## Venue defaults

`\documentclass[letterpaper, 10 pt, conference]{ieeeconf}`

| Item | Value |
|---|---|
| Body / axis labels | **10 pt** Times |
| Caption (`\footnotesize`) | 8 pt |
| Tick / legend | **8 pt** |
| `\textwidth` | 7.0 in |
| `\columnsep` | 0.2 in |
| `\columnwidth` | **3.4 in** |
| Single-column include | `width=\columnwidth` |
| Full-width include | `width=\textwidth` |

Set `figsize[0] == include_width_in` so printed pt equals matplotlib pt.

**Do not** design at 3.5 in and include at `0.9\columnwidth` (prints 10 pt labels at ~8.7 pt). Either:

- `figsize[0]=3.4` and `\includegraphics[width=\columnwidth]`, or
- `figsize[0]=0.9*3.4` and `\includegraphics[width=0.9\columnwidth]`

## Figsize recipes

| Figure | `figsize` |
|---|---|
| 3 stacked trajectories | `(3.4, 4.5)` |
| 1-panel time series | `(3.4, 1.8)` |
| Grouped bars | `(3.4, 2.5)` |

## Colors (plots + caption macros)

```latex
\definecolor{koopmanpurple}{HTML}{7030A0}
\definecolor{n4sidblue}{HTML}{4A90E2}
\definecolor{koopmanred}{HTML}{8B0000}
\definecolor{constraintred}{RGB}{255,0,0}
\definecolor{softorange}{RGB}{255,165,0}
\definecolor{desiredgreen}{RGB}{0,128,0}
```

| Role | Hex | Style |
|---|---|---|
| Measured / plant | `#000000` | solid, lw 1.2 |
| Baseline (N4SID) | `#4A90E2` | dashed, lw 1.2 |
| Proposed (Koopman linear) | `#7030A0` | dash-dot, lw **1.6** |
| Third series (Koopman NL) | `#8B0000` | dotted, lw 1.5 |
| Hard constraint | red | dashed, lw 1.0 |
| Soft / desired band | red / orange / green | `axhspan` alpha 0.15–0.2 |

Prefer **no in-figure legend**. Caption names the colors with `\textcolor`.

## Caption and TeX include

Short. Not a paragraph. Colored words, not a novel.

```latex
\begin{figure}
\centering
\includegraphics[width=\columnwidth]{figures/outputs_comparison.pdf}
\caption{Output trajectories: Koopman EMPC (\textcolor{koopmanpurple}{purple})
vs.\ N4SID EMPC (\textcolor{n4sidblue}{blue}).}
\label{fig:outputs}
\end{figure}
```

Define caption colors once to match the plot hex. Do not repeat the same figure in the text.

Ylabels: paper math + unit, upright text subscripts:

```python
r'$q_{\mathrm{feed}}$ [cm$^3$s$^{-1}$]'
```

## Matplotlib template

```python
from pathlib import Path
import matplotlib.pyplot as plt

BODY_PT = 10
TICK_PT = 8
COL_IN = 3.4          # ieeeconf \columnwidth
TEXT_IN = 7.0         # ieeeconf \textwidth
OUT = Path("paper/figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": BODY_PT,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "legend.fontsize": TICK_PT,
    "axes.titlesize": BODY_PT,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

fig, axes = plt.subplots(3, 1, figsize=(COL_IN, 4.5), sharex=True)
# ... plot in physical units, paper symbols, no title ...
axes[-1].set_xlabel(r"Time step $k$", fontsize=BODY_PT)
fig.tight_layout(pad=0.3)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "name.pdf", format="pdf")  # no bbox_inches unless you re-measure
```

- No seaborn / default 12×6 screen figsize.
- Shared x-axis on stacks; label x only on the bottom panel.
- `xlim` from 0 (small negative pad is OK). Nice y ticks (`MultipleLocator`). Do not clip data.
- Light grid only. `tight_layout(pad=0.3)` or `constrained_layout`.
- Physical / descaled units, not scaler units.

After `bbox_inches="tight"`, measure the PDF MediaBox width and set `\includegraphics[width=<that>in]` or redesign so width stays `COL_IN`.

## Paper vs presentation (do not mix)

| | Paper | Presentation |
|---|---|---|
| Font | Times New Roman | CMU Sans Serif |
| Width | 3.4 in column | ~8 in slide |
| Title | none | optional |
| Legend | usually caption-only | on the figure |
| Output | `paper/figures/*.pdf` | `presentation/*_presentation.pdf` |

## Readability checklist

- [ ] Printed axis labels ≈ 10 pt; ticks ≈ 8 pt
- [ ] Fits column; no overflow, collision, or clipped ticks
- [ ] Vector text in the PDF (selectable), not a screenshot
- [ ] x starts at 0 or a stated window; y ticks are round and not cramped
- [ ] Proposed curve is the most visible
- [ ] Notation and units match the paper
- [ ] Caption is one or two sentences
