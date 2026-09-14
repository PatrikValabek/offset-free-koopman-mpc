---
name: paper-figures-custom
description: Make publication-quality paper figures from data so fonts and axis labels print at the same size as the paper body text. Use when the user says "according to this skill", "according to this rule, make figures", "make figures for the paper", or asks for IEEE/conference matplotlib plots, captions, or figure sizing. Ask if unsure.
---

# Paper figures from data

Follow this skill whenever the user wants figures for a paper. **Ask if unsure.** Do not invent venue, symbols, units, series, or colors.

Sizes, palette, `rcParams`, and copy-paste template: [reference.md](reference.md).

## Workflow

Copy and track:

```
Figure task:
- [ ] 1. Confirm unknowns (ask, then wait)
- [ ] 2. Read the paper TeX (class, font, column, existing figures, notation)
- [ ] 3. Inspect the data (shapes, units, which series)
- [ ] 4. Design at printed size; match fonts
- [ ] 5. Draw; export vector PDF
- [ ] 6. Write a short caption + includegraphics at the designed width
- [ ] 7. Check the size math and readability
```

### 1. Ask first

Ask, then wait. Do not start plotting if any of these are unknown:

- Venue / document class / body point size / one- vs two-column
- Paper vs slides (never reuse slide styling for the paper)
- Data files and which signals to show
- Single-column (`\columnwidth`) vs full-width (`\textwidth`)
- Symbols and units as they appear in the paper
- Legend in the figure vs colors named in the caption
- Output path (default: `paper/figures/<name>.pdf`)

If this is the ECC `ieeeconf` paper and nothing contradicts it, use the IEEE 10 pt two-column defaults in [reference.md](reference.md).

### 2. Size and font law

```
printed_pt = matplotlib_pt * (include_width_in / figsize_width_in)
```

- `figsize[0]` must equal the LaTeX include width in inches.
- Axis labels = body text size (IEEE 10 pt conference: **10 pt**).
- Ticks / in-figure legend = caption size (IEEE: **8 pt**).
- Typeface = paper typeface (IEEE: Times / Times New Roman + STIX math).
- No titles on paper figures. No 12×6 exploratory figsize. No shrinking a large figure in TeX.

### 3. Draw

- Vector PDF, `pdf.fonttype=42`. No seaborn.
- Proposed method thicker than the baseline; distinct linestyles if ≥3 series.
- Constraint / desired regions as `axhspan` / dashed limits, not extra noisy lines.
- Shared-x stacks; x label only on the bottom panel.
- Paper notation in ylabels, including units. Physical / descaled units, not scaler units.
- Prefer no in-figure legend; put color names in a **short** caption with `\textcolor`.

### 4. Stop and ask

Stop and ask rather than invent a symbol, pick a non-paper color, drop a series, or change axis units.

## Done only when

- Font family matches the paper.
- Axis labels print at body size; ticks readable at ~8 pt.
- Figure fits the column without overflow or hairline labels.
- PDF is vector text (not a raster screenshot).
- Notation matches the paper.
