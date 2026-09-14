# Voice, register, and lexicon

Grounded only in the 13 papers listed in [corpus.md](corpus.md).
**[M]** mandatory · **[P]** preferred · **[O]** pick one and stay consistent.

## 1. Register

**[M]** Never first-person singular. Zero instances of "I"/"my" across all 13 papers.

**[M]** Never address the reader with a question, an exclamation, or a contraction. Write "does
not", "cannot", "it is".

**[P]** **Abstract: impersonal.** Subject is `This paper` / `This work` / `The paper`:
> This paper presents a novel deep Koopman Model Predictive Control framework for energy-intensive processes, addressing the critical challenge of real-time nonlinear control in pasteurization systems.

> The paper introduces a safe and real-time tunable approximation of the optimal controller based on a neural network (NN).

> This paper presents a library-free, approximated Model Predictive Control (MPC) for systems with fast dynamics and limited computational resources.

At most one `we` verb is tolerated in an abstract (`we propose`, `we show`).

**[P]** **Body §2 onward: first-person plural.** `we`/`our` carries methodological choices and
observations; passive carries definitions, setup, and other people's work:
> In this work, we compensate for the gap between lifted measurements and real states.
> Therefore, we propose to use Kalman filter estimation to estimate the unmeasurable lifted states.
> We leverage the deep Koopman-based identification, which is a data-driven method that ...
> The training of the Koopman model was done on a MacBook Pro with an Apple M1 Pro processor.

**[M]** Do not mix registers inside one section. Pick impersonal or first-person per section and hold it.

**[P]** Reader-address is allowed, but only through this closed set:
`Note that …` · `Note, …` · `It is important to note that …` · `As can be seen in Fig. N` ·
`It can be observed that …` · `we can see that …` · `Consider the nonlinear discrete-time system` ·
`Without loss of generality, …` · `Throughout this paper, we consider …` · `see, e.g., [x]` · `cf.`

Never `recall that`, never `one can see`, never `you`.

## 2. Sentence and paragraph texture

**[P]** Sentences 18–35 words. One main clause plus one or two participial/relative tails.
Coordinating two full independent clauses is rare.

**[P]** Paragraphs 4–8 sentences, opening with an explicit topic sentence. Sections open with a
scoping sentence:
> In this section, we present the preliminary concepts used throughout this paper.
> This section proposes a framework for data-driven identification and control using Koopman operator theory.
> In this section, we present and analyse the experimental results from the closed-loop control conducted using the laboratory PU.

**[P]** Purpose-first infinitive openers are a signature:
> To evaluate the performance of the identified models in a realistic setting, we implemented MPC closed-loop control using the PU.
> To further investigate the model accuracy, we tuned the weighting matrices …
> To achieve a fast and memory-efficient implementation, …

**[P]** Sentence-initial connectives, restricted to the ones the group actually uses:
`However,` `Moreover,` `Furthermore,` `Therefore,` `Specifically,` `Finally,` `First,` `Second,`
`Here,` `In particular,` `Additionally,` `Importantly,` `Overall,` `As expected,` `Naturally,`
`Nevertheless,` `On the other hand,` `In contrast,` `Conversely,` `Consequently,` `Subsequently,`

**[M]** No em-dash asides. Dashes appear only in captions as appositives (`u1 – flow rate of the
feed pump`) and in compounds.

**[P]** No semicolons joining independent clauses. Semicolons only separate items inside an
in-sentence list.

**[P]** Inline `(i) … (ii) … (iii)` enumeration inside a single sentence is heavily used and is the
group's default for short lists:
> The contribution of this paper can be summarised as follows: (i) Integration of the Koopman MPC framework with state observers to match lifted states with measured physical states, (ii) Experimental validation of the proposed framework using a pasteurisation unit, …

**[O]** Bulleted or numbered lists in body text: used by some papers (Neurocomputing contributions,
Koopman MPC workflow), avoided entirely by others. If used, keep them to contributions, workflow
steps, or limitation lists.

**[M]** Never end a section with a bulleted takeaway summary. Sections end on prose.

## 3. Lexicon — use these

They carry the group's voice and are all grammatically clean.

**Framing** — `the proposed approach/method/framework` · `the considered <thing>` · `In this work,
we` · `Throughout this paper, we consider` · `Without loss of generality, we consider` · `For the
purpose of this paper` · `This section presents/proposes` · `As a consequence,` · `In contrast,` ·
`see, e.g., [x]` · `cf.` · `i.e.,` · `e.g.,` · `namely` · `respectively` (heavily)

**Technical** — `memory footprint` · `computational burden` · `computational effort` · `real-time
implementation` · `the offline phase` / `the online phase` · `at each sampling instant` · `in a
receding-horizon fashion` · `boils down to` · `reduces to` · `is negligible` · `constraint
satisfaction` · `recursive feasibility` · `closed-loop stability` · `process-model mismatch` ·
`steady-state offset` · `the lifted state` · `energy-intensive`

**Evidence** — `It can be observed that` · `As can be seen in Fig. N` · `we can see that` · `A
consistent pattern is observed across all experiments:` · `This is a consequence of` · `The
improvement is due to` · `This occurs because` · `Overall, …`

**Intensity** — `significantly` is the group's main intensifier. `considerably`, `substantially`,
`negligible`, `modest` are the others. Use nothing stronger.

## 4. Never use

**[M]** These appear nowhere in the 13 papers:

`delve` · `harness` · `pivotal` · `seamless` · `cutting-edge` · `paradigm shift` · `landscape` ·
`realm` · `tapestry` · `it is worth noting that` · `in today's world` · `robustly` as a filler
adverb · `to the best of our knowledge` (the corpus uses `the author's`) · rhetorical questions ·
exclamation marks · contractions · first-person singular

**[P]** Use sparingly, at most once or twice per paper — that is their actual frequency in the
corpus: `novel` · `crucial` · `state-of-the-art` · `leverage` · `promising` · `key innovation`.

## 5. Group errors — do not reproduce

The corpus carries recognisable non-native constructions. They are part of the fingerprint but they
are **errors**, and reviewers at these venues flag them. **[M]** Avoid all of the following. If the
goal is the group's texture, deliver it through §3's phrase list instead.

| In the corpus | Write instead |
|---|---|
| `where N is prediction horizon` | `where $N$ is the prediction horizon` (dropped articles) |
| `allows to account for` | `allows the controller to account for` |
| `As the consequence,` | `As a consequence,` |
| `proper for real-time implementation` | `suitable for real-time implementation` |
| `Both, the computation time and the energy` | `Both the computation time and the energy` |
| `Note, that` | `Note that` |
| `showed, that the number` | `showed that the number` (comma before restrictive clause) |
| `increased energy efficiency in 82%` | `by 82 %` |
| `The optimization problem consist of` | `consists of` (agreement after a long subject) |
| `constrains` | `constraints` |
| `energy efficacy` | `energy efficiency` |
| `analyses` for a single analysis | `analysis` |
| `for any θ(t) ∈ Ω holds:` | `the following holds for any $\theta(t) \in \Omega$:` (verb-final order) |
