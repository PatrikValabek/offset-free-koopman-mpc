---
name: academic-writing-style
description: Write paper prose in our research group's voice and structure, distilled only from 13 of the group's own published papers. Use when the user says "according to this skill", asks to draft, rewrite, or polish any part of a paper (abstract, introduction, preliminaries, method, results, conclusions, captions), asks how the group phrases something, asks whether a convention is mandatory or breakable, or asks whether text reads as AI-generated. Ask if unsure.
---

# Group academic writing style

Every rule here is grounded in what 13 of the group's own published papers actually do. No outside
style authority. **Ask rather than invent** a claim, a number, a citation, or a symbol.

- Section-by-section conventions (skeleton, abstract, intro, equations, floats, results, conclusions, citations, spelling): [reference.md](reference.md)
- Voice, register, phrase lexicon, verbatim sample sentences, forbidden words, group errors: [voice.md](voice.md)
- Which 13 papers, and where they disagree with each other: [corpus.md](corpus.md)
- Symbol definitions and the `where` clause: owned by `.cursor/rules/latex-paper-writing.mdc`

Tags used in the reference files: **[M]** mandatory (violated in 0–1 of 13 papers) ·
**[P]** preferred, break only with a reason · **[O]** genuine variation, pick one and stay consistent.

## Workflow

Copy and track:

```
Writing task:
- [ ] 1. Read \documentclass -> journal or conference mode
- [ ] 2. Read the surrounding TeX (notation, spelling line, % spacing, citation style)
- [ ] 3. Confirm unknowns (ask, then wait)
- [ ] 4. Draft against the non-negotiables below
- [ ] 5. Sweep the "Done only when" checklist
```

### 1. Venue check

`elsarticle` / journal `IEEEtran` → **journal mode** (this project: `elsarticle`, `authoryear`).
Conference class, `Abstract—`, `Index Terms—` → **conference mode** (6 pages, roman sections,
numeric citations, funding in a page-1 footnote).

### 2. Voice anchors

Model the prose on the two flagship journal papers: *Experimental validation of deep Koopman MPC
for real-time pasteurization unit control* (Valábek, Horváthová, Pannocchia, Klaučo, Control Eng.
Practice 2026) and *Supervised learning for robust predictive control* (Horváthová, Kiš, Klaučo,
Oravec, Neurocomputing 2026). The earlier Valábek conference papers are the group's weaker
register — do not use them as the prose model.

### 3. Non-negotiables

1. No first-person singular, ever. No rhetorical questions, exclamation marks, or contractions.
2. Abstract carries no citations, no roadmap, and no future work.
3. One register per section. Impersonal or first-person plural, held consistently.
4. Every prior method in the introduction gets **two** sentences: one describing it, one limiting it.
5. A display equation is a constituent of a sentence, never free-standing. It ends in a comma when
   a `where` clause follows, otherwise a period.
6. No section or subsection ends on a display equation. Minimum tail is the `where` clause.
7. Cross-reference equations as bare `(8)` or `the MPC problem (8)`. Never `Eq. (8)`.
8. `Fig. N` and `Table N` in prose. Every float discussed in prose; every colour and line style
   decoded in the caption.
9. No timing or memory number without the hardware named. Every solver and toolbox named.
10. Units take a space (`10 ms`, `50 °C`). Percent spacing consistent document-wide.
11. Conclusions introduce no new figures, tables, or citations. The last sentence is future work.
12. Self-citations in the third person. `To the best of the author's knowledge`, never `our`.
13. Headings in sentence case. Acronyms expanded once, at first use only.
14. Never use the words in [voice.md](voice.md) §"Never use" — they appear nowhere in the corpus.
15. Never reproduce the non-native constructions in [voice.md](voice.md) §"Group errors".

### 4. Stop and ask

Stop and ask rather than invent a numerical result, attribute a claim to an uncited source, name a
solver or hardware that was not stated, coin a symbol not already in the paper, or assert novelty
that has not been established.

## Done only when

- Venue mode identified and applied consistently.
- Every mandatory item above holds.
- Contributions are checkable against specific sections.
- At least one honest concession to the baseline appears in the results.
- Spelling internally consistent; the corpus is split, so match whatever the document already uses.
- No acronym re-expanded; no orphan float; no section ending on an equation.
- The [reference.md](reference.md) pre-submission checklist is swept.
