# MEMORY LAB — Understanding Linear Attention

> **DataForge 2026: Pathway Track — "Explain the Frontier"**

## Central Educational Claim

**Linear attention reorganizes attention so that contextual memory can be updated incrementally rather than explicitly comparing every pair of tokens.**

Memory Lab is an interactive, browser-based educational explainer using transparent toy computations and animations to teach kernelized linear attention, its fixed-shape recurrent state, and the associated memory/precision trade-offs.

## Target Audience & Learning Objectives

**Audience:** AI/ML students, data scientists, ML engineers, researchers, and technical judges.

**Prerequisites:** Basic linear algebra (vectors, matrices, dot/outer products), Transformer attention, and autoregressive KV caching.

Learners should be able to explain softmax versus kernel attention, compare N×N interactions with sequential updates, interpret the recurrent state as associative fast-weight memory, relate outer-product writes to Hebbian/fast-weight ideas, explain the BDH/BDH-CQ connection without claiming reproduction, and identify retrieval/interference and hardware-runtime trade-offs.

## Interactive Substrate

- **Hero Memory Engine:** animated `INPUT → φ(k) → ΔS → STATE → QUERY` flow.
- **Pairwise Footprint Visualizer:** live JavaScript rendering for N=4–100.
- **100-Token Race:** compares N² pairwise slots with N sequential updates.
- **Core Memory Flow Experiment:** live d=4 toy linear algebra with step-by-step and auto-run modes.
- **State Inspector & Query Probe:** live state/readout inspection.
- **Prediction Challenge:** learner prediction followed by computed comparison.
- **BDH / BDH-CQ tabs:** conceptual/formal research connection, explicitly not a reproduction of the full architecture.
- **60-Second Explainer:** short synthesis activity.

### Evidence Discipline
The interface distinguishes `[RESEARCH EVIDENCE]`, `[FOUNDATIONAL / BACKGROUND]`, `[BDH TECHNICAL REPORT]`, `[CONCEPTUAL RELATION]`, and `[TOY COMPUTATION]`. The toy JavaScript substrate is not presented as a benchmark or production LLM implementation.

## Mathematical Core

Standard softmax attention:

$$\mathrm{Attn}(Q,K,V)=\mathrm{Softmax}\left(\frac{QK^T}{\sqrt d}\right)V$$

A kernelized formulation replaces the similarity with a feature-map inner product φ(q)ᵀφ(k), allowing the sequence sum to be regrouped:

$$S_t=S_{t-1}+\phi(k_t)v_t^T$$
$$z_t=z_{t-1}+\phi(k_t)$$
$$y_t=\frac{\phi(q_t)^TS_t}{\phi(q_t)^Tz_t}$$

The key educational point is that the recurrent state has fixed shape with respect to sequence length. This is a memory-scaling statement for the recurrent representation, not a claim that every implementation is always faster than optimized softmax attention.

## BDH / BDH-CQ Connection

The project uses Pathway's Dragon Hatchling (BDH) and BDH-CQ research as a conceptual frontier connection: synaptic/fast-weight-style recurrence, sparse non-negative activations, and additive in-state demonstration memory. **Memory Lab does not reproduce or benchmark BDH/BDH-CQ.** Simplified diagrams and equations are pedagogical abstractions and are labeled accordingly.

## Known Limitations

1. A fixed-size associative state has finite capacity and can suffer interference as context grows.
2. Kernel similarity does not reproduce the exact sharpness of softmax attention.
3. Asymptotic complexity does not guarantee lower wall-clock latency on every GPU/workload; optimized softmax attention can be faster for some regimes.
4. The browser demo uses a tiny d=4 toy feature space for transparency and is not representative of production model dimensions.

## Architecture

```text
linear-attention-explainer/
├── index.html
├── style.css
├── script.js
└── README.md
```

The artifact is a static HTML/CSS/JavaScript application with no backend, database, package manager, or build step.

## How to Run Locally / Reproduce

No installation is required.

1. Clone or download the repository.
2. Open `index.html` in a modern browser, or serve the directory with `python -m http.server 8000`.
3. Interact with the sequence-length sliders, memory-flow controls, state inspector, prediction challenge, and explainer timer.

The numerical demonstrations are computed client-side by `script.js`; values are educational toy calculations rather than benchmark results.

## Public Artifact & Repository

- **Live interactive demo:** https://srko7596.github.io/linear-attention-explainer/
- **Public repository:** https://github.com/srko7596/linear-attention-explainer

## Submission Materials

The submission package contains:
- `ONE_PAGE_CONCEPT_SUMMARY.pdf` — one-page concept summary.
- `BLOG.pdf` — separate educational blog PDF.
- `SOURCES_AND_LICENSES.md` — source, asset, and license record.
- `AI_DISCLOSURE.md` — AI assistance/disclosure record.
- `LICENSE.txt` — MIT License.

## Sources

Primary/recent references used for the technical framing include:
- Zhen et al., **COSFORMER: Rethinking Softmax in Attention**, ICLR 2022.
- Sun et al., **Retentive Network: A Successor to Transformer for Large Language Models**, 2023.
- Yang et al., **Gated Linear Attention Transformers with Hardware-Efficient Training**, 2024.

Foundational context includes Katharopoulos et al. (2020) and Schlag et al. (2021). Pathway's BDH/BDH-CQ technical reports are used for the explicitly labeled BDH connection.

## Source / License / AI Disclosure

- **Code:** original project code for this educational artifact; MIT licensed.
- **KaTeX:** used for mathematical rendering under its MIT license.
- **Google Fonts:** Inter and JetBrains Mono, under their respective SIL Open Font License terms.
- **Graphics:** custom SVG/Canvas elements created for this project.
- **AI assistance:** AI tools assisted with architectural ideation, pedagogical structuring, formula formatting, and code drafting. The final mathematical framing, claims, sources, and project behavior were reviewed for scientific accuracy and disclosed here rather than presented as independently authored AI-free work.

## License

MIT License. See `LICENSE.txt` in the submission package.
