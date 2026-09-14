# Corpus provenance

Every rule in `SKILL.md`, `reference.md`, and `voice.md` was derived from these 13 papers and
nothing else. The PDFs are not kept in the repository; re-download from the publisher if a rule
needs to be re-derived, verified, or extended.

Metadata below was read off the manuscripts themselves. Where a field was not printed on the
manuscript it is marked as such rather than guessed.

## Journal papers (9)

| # | Authors | Title | Venue |
|---|---|---|---|
| J1 | Valábek, Horváthová, Pannocchia, Klaučo | Experimental validation of deep Koopman MPC for real-time pasteurization unit control | Control Engineering Practice, in press (typeset Nov 2025) |
| J2 | Horváthová, Kiš, Klaučo, Oravec | Supervised learning for robust predictive control: Safe and tunable approach | Neurocomputing 671 (2026) 132637 |
| J3 | Oravec, Klaučo | Real-time tunable approximated explicit MPC (Technical communique) | Automatica 142 (2022) 110315 |
| J4 | Kohút, Klaučo, Kvasnica | Unified carbon emissions and market prices forecasts of the power grid | Applied Energy (2024) |
| J5 | Oravec, Horváthová, Bakošová | Energy efficient convex-lifting-based robust control of a heat exchanger | Energy (2020) |
| J6 | Klaučo, Kalúz, Kvasnica | Machine learning-based warm starting of active set methods in embedded model predictive control | Engineering Applications of Artificial Intelligence 77 (2019) 1–8 |
| J7 | Drgoňa, Kiš, Tuor, Vrabie, Klaučo | Differentiable predictive control: Deep learning alternative to explicit model predictive control for unknown nonlinear systems | Journal of Process Control 116 (2022) 80–92 |
| J8 | Klaučo, Kalúz, Kvasnica | Real-time implementation of an explicit MPC-based reference governor for control of a magnetic levitation system | Control Engineering Practice (2017) |
| J9 | Dyrska, Horváthová, Bakaráč, Mönnigmann, Oravec | Heat exchanger control using model predictive control with constraint removal | Applied Thermal Engineering 227 (2023) 120366 |

## Conference papers (4)

Venue and year are not printed on these manuscripts.

| # | Authors | Title |
|---|---|---|
| C1 | Valábek, Horváthová, Klaučo | Deep Koopman Economic Model Predictive Control of a Pasteurisation Unit |
| C2 | Valábek, Wadinger, Kvasnica, Klaučo | Deep Dictionary-Free Method for Identifying Linear Model of Nonlinear System with Input Delay |
| C3 | Dzurková, Valábek, Mészáros, Kalúz, Klaučo | Approximated Explicit NMPC via Reinforcement Learning for Homomorphically Encrypted Process Control |
| C4 | Horváthová, Jiang, Holaza, Olaru, Oravec | Variance-Adaptive Approximated Model Predictive Control |

## Weighting

**J1** and **J2** are the voice anchors — model new prose on them. **C1**–**C3** are the earlier
Valábek conference papers and represent the group's weaker register; they were used for structural
evidence (conference skeleton, numeric citations, page-1 funding footnote) but not as a prose model.

## Known inconsistencies inside the corpus

Documented rather than averaged away, because they are the reason several rules are `[O]` instead
of `[M]`:

- **Spelling.** American in J6, J7, J8; British in J2 and J5; mixed in J1 (`behaviour`, `analysed`,
  `colour` alongside `pasteurization`, `utilizing`).
- **Percent spacing.** Spaced (`30 %`, `94.4 %`) in J1 and J2; unspaced in J4 and J9; J2 itself
  mixes (`99.5 %` but `0.74%`).
- **Theorem environments.** Heavy in J2 and J3, entirely absent from J1 and J4.
- **Roadmap paragraph.** Present in J1, J2, J5, J9 and in C1/C3; absent from J3, J6, J7, J8.
- **`we` in the abstract.** Absent from J1, J5, J8 and all four conference abstracts; present once
  in J2, J3, J6, J7, J9.
- **Caption content.** Contents-only in J1, J2, J9; states conclusions in J4.
