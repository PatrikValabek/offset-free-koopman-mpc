# Structural conventions

Grounded only in the 13 papers listed in [corpus.md](corpus.md).
**[M]** mandatory · **[P]** preferred · **[O]** pick one and stay consistent. `⟨conf⟩` = conference only.

## 1. Document skeleton

**[P]** Journal skeleton, following the Koopman MPC paper (use this for the offset-free follow-up):

```
1. Introduction
2. Preliminaries                    (2.1 theory  2.2 baseline method  2.3 MPC framework)
3. <Proposed framework>             (3.1 identification  3.2 the new mechanism  3.3 the controller)
4. <Plant>                          (physical description, variables named u1..u3 / y1..y3)
5. Model of the <plant>             (5.1 baseline model  5.2 proposed model  5.3 open-loop performance)
6. Control setup
7. Results and discussion
8. Conclusions
CRediT authorship contribution statement
Declaration of competing interest
Acknowledgments
References
```

`⟨conf⟩` 6 pages, roman-numeral sections, `I. INTRODUCTION` … `VIII. CONCLUSIONS`; funding in an
unnumbered footnote on page 1, not a section.

**[M]** Headings in sentence case (`Model of the pasteurization unit`). Only the first word and
proper nouns/acronyms capitalised.

**[P]** Journal back matter in this order: CRediT statement, Declaration of competing interest,
Acknowledgments. If a language model was used, add Elsevier's *Declaration of generative AI …*
statement (the Neurocomputing paper carries one).

**[P]** Funding template (reuse verbatim, swap grant numbers):
> The authors M. Klaučo and P. Valábek gratefully acknowledge the contribution of the Scientific Grant Agency of the Slovak Republic under the grants VEGA 1/0239/24, the Slovak Research and Development Agency under the project APVV-20-0261.

**[O]** Highlights (Neurocomputing style): 5 items, one short clause each, the last two carrying the
headline numbers:
> • Lightweight, library-free controller design suitable for embedded hardware use.
> • Case study 2: Quadrotor control, 99.5% less memory than explicit MPC.

**[M]** No nomenclature table in the recent papers; symbols are defined inline in `where` clauses.
(Only the two heat-exchanger papers carry a nomenclature list.)

## 2. Abstract

**[M]** No citations. Zero across all 13 papers.

**[M]** No roadmap ("the paper is organised as…") and no future work.

**[P]** 7–11 sentences (journal), 5–10 `⟨conf⟩`. Fixed rhetorical order:
1. Contribution + object, or problem framing.
2. Mechanism — how the method works, one sentence.
3. The specific novelty, flagged (`A key innovation is …`).
4. Validation setting.
5. Headline quantitative result against a **named** baseline.
6. Why it matters practically.
7. Positioning claim, if any (`This work represents one of the first experimental implementations of …`).

**[P]** At most two numbers, each tied to its baseline:
> Experimental validation on a laboratory-scale pasteurization unit demonstrates that the proposed deep Koopman MPC framework achieves 30 % improvement in control performance compared to conventional subspace identification methods, while maintaining real-time execution within 10 ms on standard hardware.

## 3. Introduction

**[P]** Open on a broad field or industrial premise, uncited in the first sentence, then funnel:
> Pasteurization is an important part of modern food and beverage production, which ensures microbial safety and extends shelf life without compromising product quality.
> Tunable control laws are applicable in a wide range of control scenarios, not only in industrial settings [1] but also in robotics and other fields [2].

**[M]** **The gap engine.** Every prior method gets exactly two moves: one sentence describing it,
one sentence limiting it. This is the single most consistent structure in the corpus:
> The work of Riverol et al. (2008) uses a neural network to approximate the nonlinear model of a pasteurization unit, but this model is used only in fuzzy controllers.
> Subspace identification methods have also been applied to PUs to compute linear state-space models directly from data (Dzurková et al., 2024), enabling efficient MPC design. Their key limitation is the assumption of linearity, which can lead to poor performance in systems with significant nonlinear dynamics …
> However, heuristic fuzzy controllers use a model that is based on expert-defined rules that lack any process modeling needed for consistent and safe thermal control.
> Although lightweight, this sampling strategy lacks a mechanism to guide exploration of the solution space, making it inefficient for multi-input or high-dimensional systems.

Limitation connectives actually used: `However,` · `Their key limitation is …` · `The main drawback
… lies in …` · `This, however, does not take into account …` · `While powerful, this approach does
not guarantee …` · `Although lightweight, …` · `but a fully robust solution remains an open challenge`.

**[P]** State the gap you fill in one sentence immediately before the contributions:
> As of our knowledge, a practical chemical engineering or energy-intensive application of deep Koopman MPC has not yet been published.

**[P]** Novelty claim, if made, in the third person: `To the best of the author's knowledge, …`
Never `to the best of our knowledge`.

**[O]** Contributions as (a) prose with inline `(i)(ii)(iii)`, or (b) a bulleted list under a
`1.1 Main contributions` heading. Both occur; prose+inline is the Koopman-paper form. Each
contribution must be concrete enough to check against a section:
> The contribution of this paper can be summarised as follows: (i) Integration of the Koopman MPC framework with state observers to match lifted states with measured physical states, (ii) Experimental validation of the proposed framework using a pasteurisation unit, …

**[P]** Close the introduction with a one-paragraph roadmap, one clause per section:
> The paper begins with Section 2, introducing the Koopman theory, MPC framework, and subspace identification. Section 3 presents the proposed deep Koopman MPC framework, detailing the data-driven model identification, lifted state correction, and implementation workflow. … The paper concludes in Section 8 with a summary of key contributions.

## 4. Equations and mathematical prose

Symbol definitions and the `where` clause: `.cursor/rules/latex-paper-writing.mdc`. Grammar here.

**[M]** A display equation is a **constituent of a sentence**, never free-standing. The lead-in is
an incomplete clause that the equation completes:
> Considering the nonlinear discrete dynamical system:
> As a result of the identification process, we obtain the following linear representation of the deep Koopman model:
> To evaluate the closed-loop performance from experimental data, we compute the cumulative cost as:
> Following the approach of the tube-based MPC design [20], the optimisation problem has the form
> The controller is constructed as follows:

Reusable stems: `…of the form` · `…given by` · `…is defined as` · `…as follows:` · `…can be written
as` · `…read` · `…is formulated as … of the form` · `…resulting in`.

**[O]** Lead-in punctuation: colon when the lead-in is a complete clause (`is defined as:`, `as
follows:`); nothing when the equation is the grammatical object (`of the form`, `given by`). Pick
one convention per document.

**[M]** Display equations carry terminal punctuation: a **comma** when a `where` clause follows, a
**period** when the sentence ends there.

```latex
% ✅ GOOD
z_{k+1} = A z_k + B u_k, \label{eq:koop}
% ... where $z_k \in \mathbb{R}^{n_z}$ is the lifted state, ...

% ✅ GOOD (sentence ends)
Q_\text{u} = \diag([1, 1, 1]).

% ❌ BAD — no terminal punctuation
z_{k+1} = A z_k + B u_k
```

**[M]** **No section or subsection ends on a bare display equation.** Verified across 12 of 13
papers. Minimum tail is the `where` clause; the norm is at least one sentence of interpretation.

**[P]** Number every display equation; sub-letter every line of an optimisation block so individual
constraints can be cited: `(8a)`–`(8e)`.

**[M]** Cross-reference equations as bare parenthesised numbers, optionally appended to a noun
phrase. Never `Eq. (8)` or `Equation (8)` in running prose:
> the MPC problem (8) · the constraint (4e) · as outlined in Section 2.3 · solving the optimization problem either using the Koopman-based model (8) or the N4SID model (6)

**[P]** After an optimisation block, walk the reader through the cost term by term, then the
constraints, then the tuning matrices — in printed order:
> The first term represents … The second and the third term penalize … Finally, (9g) represents a move blocking constraint which …

**[O]** Theorem-like environments: `Assumption N.` / `Remark N (Named Thing).` / `Definition N.` /
`Lemma N.` / `Proposition N.` / `Proof … □`. Heavily used in the theory-forward papers
(Neurocomputing, Automatica), entirely absent from the applied ones (Koopman MPC, Applied Energy).
Environment names take Title Case even though headings are sentence case. Cite the source in the
label when the result is not yours: `Lemma 3.1 (Stability [14]).`

**[P]** Algorithms: `Require:`/`Ensure:` or `Input:`/`Output:` headers, numbered steps, `←` for
assignment, terse `//` comments, steps cross-referencing equations (`per (25)–(26)`). Refer to them
articleless: "Algorithm 1 takes …", "as summarized in Algorithm 2".

## 5. Figures and tables

**[M]** In prose: `Fig. N` (abbreviated) and `Table N`. Never `Figure N` in journal mode.
`⟨conf⟩` roman table numerals: `Table I`.

**[M]** Every figure and every table explicitly discussed in prose. No orphan floats.

**[M]** Caption anatomy: sentence case, terminal period, opening **verbless noun phrase** naming
what is shown, then a sentence per panel/trace decoding **every** line style and colour:
> Fig. 7. Control results for the controlled variables of the CS #3 ↓. The blue line is for the MPC using the N4SID model, and the red line is for the MPC using the deep Koopman model. The Black dashed line is the setpoint. The light green area indicates the ±1 °C tolerance band around the setpoint, which is considered the steady state once the response remains within it.

> Fig. 4. Scheme of experimental device - pasteurization unit. The variables in the scheme have the following meaning: u1 - flow rate of the feed pump, u2 - flow rate of the hot water pump, u3 - the power to the heating spiral, y1 - the temperature at the end of a holding tube T1, …

**[P]** Captions describe contents, not conclusions. Verdicts belong in prose.

**[P]** Table captions state the normalisation convention:
> Comparison of models (normalized as 100 % stands for the performance of the N4SID model).

**[O]** Bolding the best value: the group mostly does **not** bold; it either normalises the
baseline to 100 % or names the winner in prose. If you highlight, declare it in the caption.

**[P]** Describe a results plot as: what it shows → the qualitative behaviour → attribution to a
specific equation or design choice:
> A naive initialisation z_k = g(y_k) … neglects historical information and amplifies the impact of process-model mismatch and noise, resulting in steady-state offsets and a significant decrease in control performance. On the other hand, incorporating the Kalman filter correction, as we propose, removes the offset and restores accurate setpoint tracking.

## 6. Results and numbers

**[M]** Never report a computation time or memory figure without naming the hardware:
> The timing was performed on an Apple M1 Pro processor and includes the time required for matrix construction, optimization, and total execution.
> The proposed approach was designed and implemented using MATLAB R2021b on a PC with an i5 CPU (2.7 GHz) and 8 GB RAM.

**[M]** Name every tool and solver: `The MPC optimization problem was solved using the Gurobi
solver.` · `The optimisation problems were formulated and solved using MPT [36] and Yalmip [37]
toolboxes.`

**[P]** State the fair-comparison protocol explicitly:
> To ensure a fair comparison of the control performance across models, the control configuration was kept consistent across both obtained models. The only variation was the underlying model (deep Koopman or N4SID) and the corresponding state estimator.

**[P]** Report averages **and** worst case for timing; report offline construction cost alongside
online cost.

**[P]** Percentages: the two flagship 2026 papers write a **thin space** before `%` (`30 %`,
`44.5 %`, `94.4 %`, `99.5 %`). `siunitx` is already loaded — use `\SI{30}{\percent}`. Do not mix
`30 %` and `30%` in one document.

**[M]** Units take a space: `10 ms`, `50 °C`, `2.7 GHz`, `8 GB`, `1 s`. In table headers put units
in square brackets: `OCT [ms]`, `Memory Footprint [kB]`.

**[P]** Concede to the baseline where it wins, and explain why. Strong group habit, buys credibility:
> the increase in energy consumption is negligible - typically less than 1 %, or even reduced in certain cases
> As expected, the offline construction of matrices for the deep Koopman model is slightly more computationally intensive than for the N4SID model due to its higher state dimensionality.

**[P]** Define every metric where first used, with its physical meaning and what a bad value implies
operationally:
> The second metric is the Mean Integral Absolute Error (MAE), which reflects the control performance. … A higher MAE in a control setup indicates underperforming temperature control. This may result in insufficient heating, which fails to eliminate pathogenic bacteria, or excessive heating, which can degrade essential bioactive compounds.

**[P]** Numbers hedge with `approximately`, `around`, `about`, `up to`, `almost`, `by a factor of`,
`averagely reduced by`. Bare superlatives without a number do not appear.

## 7. Conclusions and future work

**[P]** One to three paragraphs, 7–12 sentences. Recapitulate the method as a chain of decisions,
then close on the hardest defensible numbers.

**[M]** No new figures, tables, experiments, or citations.

**[M]** The **last** sentence (or last two) is future work, one concrete scoped item:
> Future work will focus on developing an offset-free Koopman Model Predictive Control framework. This will involve addressing steady-state offset issues by integrating disturbance modeling or state estimation techniques, ensuring improved tracking performance and robustness in practical applications.

**[P]** Restate one to three numbers, naming the baseline. `⟨conf⟩` the Valábek conference
conclusions avoid `we` entirely (`This paper presents …`); journal conclusions use it (`In this
paper, we presented …`).

**[O]** A separate `Discussion` subsection or `Limitations and future work` section before the
conclusions — used by Neurocomputing (§6.4) and the JPC paper (§5), organising limitations by
perspective, each with a cited remedy.

## 8. Citations

**[M]** Zero citations in the abstract.

**[M]** Self-citation in the third person, always. The corpus never writes "our previous work", "in
our earlier paper", or "we have previously shown". A self-citation is formally indistinguishable
from a stranger's:
> a poor choice of lifted function can lead to poor performance (Valábek et al., 2025)
> according to principles introduced in [10]

**[P]** Journal mode (`authoryear`, `elsarticle-harv`) — mix integrated and parenthetical forms:
> Anang et al. (2016) highlight the superior performance of MPC over the classical cascade control …
> Further development led to the use of nonlinear models for the MPC design, as shown in Thostrup et al. (2022).
> The Koopman operator theory was initially introduced by Koopman (1931), its use in control has only gained popularity in recent decades, particularly following the work of Mezić (2005).
> Therefore, accurate identification and effective control of pasteurization processes are important (Martin et al., 2018).

`⟨conf⟩` numeric `[n]`; the conference papers never put author surnames in body text, using `the
authors in [3]`, `[5] introduced an auxiliary feedback controller`, `introduced by [3] and
popularized by [4]` instead.

**[P]** One reference per claim. Bundle two or three only for a family of methods:
`(Brunton et al., 2016; Proctor et al., 2016; Williams et al., 2015)`.

**[P]** Deferral sentence — a genuine group fingerprint, reuse it:
> Further technical details regarding the model of the quadrotor can be found in [40,41].
> Further technical details regarding the experimental identification of the plant are discussed in [29].

**[P]** Cite tools and datasets like literature: `the System Identification Toolbox in MATLAB,
version 2024b (The MathWorks, Inc., 2024)`.

## 9. Spelling, hyphenation, acronyms

**[M]** No spelling mandate — the corpus is genuinely split (American in the older Klaučo/Oravec
papers, British in Neurocomputing 2026 and the 2020 *Energy* paper, mixed in the Koopman MPC paper:
British `behaviour`/`analysed`/`summarised`/`colour` alongside `pasteurization`/`utilizing`).
**Requirement: internal consistency within a document.** If the paper already contains
`linearisation` or `behaviour`, keep the British `-our`/`-ise` line throughout.

**[P]** Hyphenation, as the group actually writes it:
`closed-loop` (always, as modifier) · `real-time` (attributive) / `in real time` (adverbial) ·
`offset-free` · `data-driven` · `state-space` · `steady-state` · `setpoint` (one word) ·
`offline` / `online` (solid in the recent papers) · `energy-intensive` · `laboratory-scale`

**[P]** Stacked hyphenated compound modifiers are a visible group habit and safe to imitate:
`convex-lifting-based`, `deep-Koopman-based`, `neural-network-based`, `constraint-removal-based`,
`min-max-like`, `dictionary-free`, `library-free`, `solver-free`.

**[P]** Acronyms: lowercase expansion, acronym in parentheses, at first use only:
> Model predictive control (MPC) · a laboratory-scaled pasteurization unit (PU) · the mean-squared error (MSE) · a multiple-input-multiple-output (MIMO) dynamic process

**[M]** Do **not** re-expand an acronym in later sections. The corpus does this repeatedly (PU four
times, DDPG three times) and it is a flaw, not a feature.

## 10. Pre-submission checklist

- [ ] Venue identified from `\documentclass`; journal/conference conventions applied consistently.
- [ ] One register per section; no first-person singular anywhere.
- [ ] Abstract: no citations, no roadmap, ≤ 2 numbers, each with a named baseline.
- [ ] Every prior method in the introduction has a description sentence **and** a limitation sentence.
- [ ] Contributions are checkable against specific sections; roadmap paragraph closes the introduction.
- [ ] Every display equation is grammatically part of a sentence, carries terminal punctuation, and is numbered.
- [ ] No section or subsection ends on a display equation.
- [ ] Equations cross-referenced as `(8)`, never `Eq. (8)`.
- [ ] Every figure and table referenced in prose; every colour and line style decoded in the caption.
- [ ] No timing or memory number without its hardware; every solver and toolbox named.
- [ ] Fair-comparison protocol stated; at least one honest concession to the baseline.
- [ ] Percentages and units spaced consistently; `siunitx` used.
- [ ] Conclusions: no new material, hardest numbers restated, one concrete future-work sentence last.
- [ ] Self-citations phrased in the third person.
- [ ] Spelling internally consistent; no acronym re-expanded.
- [ ] `voice.md` §"Group errors" list swept.
- [ ] Agent-drafted prose is wrapped in `\generatedstyle` (`HTML 1B5E20` and bold); default `green` is not used.
- [ ] Agent-drafted blocks also carry `\genopen`/`\genclose` with the model name and settings that
      generated them; a faster model is not used to save time.
