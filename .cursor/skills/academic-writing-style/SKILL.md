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
- [ ] 5. Raise every remaining open question twice: in chat and as a `\todo{}` in the document
- [ ] 6. Sweep the "Done only when" checklist
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
16. Drafted prose is wrapped in `\generatedstyle` (colour `generated`, `HTML 1B5E20`, plus bold). Never the default `green`.
    Beside the colour, mark the model and settings that generated the passage.

### 4. Mark generated prose

**[M]** Every sentence, caption, or equation block drafted by the agent is wrapped in
`\generatedstyle` (colour `generated`, `HTML 1B5E20`, forest green, and bold). It must stay
readable on a printout and still be distinguishable from the author's black text. Never use the
default LaTeX `green`, which washes out in print. Do not reuse `generated` for plot traces: those
keep `parsimk`, `koopmanc`, `deepkoopman`, and `koopmanb`.

**[M]** Beside the colour, add the model name and settings that generated the text, with a small
label at the beginning and at the end of the added block (`\genopen` / `\genclose`). Record the
model that actually produced the passage. Do not substitute a faster or cheaper model to reduce
latency; generated-prose quality is not to be traded for speed. If the user names a label, use
that string verbatim.

```latex
\definecolor{generated}{HTML}{1B5E20}
\newcommand{\generatedstyle}{\color{generated}\bfseries\boldmath}
\newcommand{\textgenerated}[1]{\textcolor{generated}{\bfseries\boldmath #1}}
\newcommand{\gensource}{grok4.6high}
\newcommand{\genopen}{{\scriptsize\generatedstyle[\gensource]}}
\newcommand{\genclose}{{\scriptsize\generatedstyle[/\gensource]}}
{\generatedstyle
\genopen
... drafted section ...
\genclose
}
\caption{\textgenerated{\genopen ... drafted caption ... \genclose}}
```

Unresolved gaps stay `\todo{}` in red. Once the author accepts a passage, the `\generatedstyle`
wrapper and the model labels are removed and the text reverts to black.

### 5. Stop and ask, and leave a `\todo{}`

Stop and ask rather than invent a numerical result, attribute a claim to an uncited source, name a
solver or hardware that was not stated, coin a symbol not already in the paper, or assert novelty
that has not been established.

**[M]** Every open question is raised **twice**: once in chat, and once in the document as a
`\todo{}` at the exact spot it concerns. Chat messages scroll away; the paper is what the author
reads next. Never do only one of the two.

The `\todo{}` is self-contained: it names what is missing, why it blocks the text, and what a
usable answer looks like. It never says "see chat", never just repeats the surrounding prose, and
never restates a decision that has already been made.

```latex
% ✅ GOOD — states the gap, the reason, and the shape of the answer
\todo{This sentence characterises other authors' formulations but carries no citation, and a claim
may not be attributed to an uncited source. Which papers augment the lifted state with the
measurements? Supply the keys and they will be cited here, one reference per claim.}

% ✅ GOOD — a number that exists in the code but was never confirmed for the paper
\todo{The implementation uses $Q_\text{z} + 10^{-8} I$. State this value explicitly here, or keep
the wording qualitative?}

% ❌ BAD — no content, no question
\todo{fix this}

% ❌ BAD — defers to a channel the author is not reading
\todo{See the chat for the notation question.}
```

Placement: immediately after the sentence, `where` clause, or `\label{}` in question, on its own
line. Not inside a `\section{}` or `\caption{}` argument, which would push the red text into the
heading and the table of contents.

Remove the `\todo{}` in the same edit that resolves it. A `\todo{}` restating a settled decision is
noise, and an unresolved question with no `\todo{}` is a silent invention waiting to be published.

## Done only when

- Every open question exists both in chat and as a `\todo{}` in the document; every resolved one has
  had its `\todo{}` removed.
- Venue mode identified and applied consistently.
- Every mandatory item above holds.
- Contributions are checkable against specific sections.
- At least one honest concession to the baseline appears in the results.
- Spelling internally consistent; the corpus is split, so match whatever the document already uses.
- No acronym re-expanded; no orphan float; no section ending on an equation.
- Agent-drafted prose carries `\generatedstyle` (green and bold) and `\genopen`/`\genclose` with the model
  and settings; a faster model is not used to save time.
- The [reference.md](reference.md) pre-submission checklist is swept.
