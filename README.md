# MEMORY LAB — Understanding Linear Attention

> **DataForge 2026 — Pathway Track: Explain the Frontier**  
> Interactive educational explainer on **Linear Attention**, with a carefully scoped connection to Pathway's **Dragon Hatchling (BDH) / BDH-CQ** research.

## 1. Project Claim

**Central claim:**

> Linear attention reorganizes attention so that contextual information can be accumulated incrementally in a recurrent state, avoiding explicit materialization of the full token-to-token attention score matrix in the toy formulation presented here.

Memory Lab makes this idea manipulable rather than purely descriptive. The learner changes sequence length, observes the scaling of a simplified pairwise attention view, then steps through a deterministic toy linear-attention update and inspects the resulting state.

The demo is an **educational toy substrate**. It is not a reproduction of a production BDH, BDH-CQ, or large language model run.

---

## 2. Intended Audience

**Primary audience:**
- AI/ML students and engineers
- Data scientists
- Technical hackathon judges
- Readers with basic familiarity with Transformers

**Prerequisites:** basic matrix multiplication, vectors/outer products, and the idea of query/key/value attention.

**Target time to first insight:** approximately 60–90 seconds.

---

## 3. Learning Objectives

After using the explainer, a learner should be able to:

1. Explain why standard attention is represented by an `N × N` attention score matrix.
2. Explain the feature-map idea behind a kernelized linear-attention formulation.
3. Trace the incremental state update
   `S_t = S_(t-1) + φ(k_t) v_t^T`.
4. Distinguish the sequence-length scaling of an explicit attention score matrix from the storage growth of an autoregressive KV cache.
5. Explain why a fixed-shape recurrent state introduces memory-capacity and retrieval trade-offs rather than perfect lossless memory.
6. State the relationship to BDH/BDH-CQ without treating BDH as identical to generic Linear Attention or to Mamba-style SSMs.

---

## 4. What the Interactive Demo Does

Memory Lab is organized as a short learning journey:

`Observe → Change N → Predict → Run → Inspect State → Connect to Research → Examine Limitations`

### Core interactions

- **Hero memory engine:** a deterministic token sequence travels through `INPUT → TRANSFORM φ(k) → UPDATE ΔS → STATE S → QUERY y`.
- **Pairwise footprint:** small `N` values can be inspected directly; larger values use a Canvas heatmap.
- **100-token race:** change `N` from 4 to 100 and compare a simplified `N²` pairwise-slot view with `N` incremental state updates.
- **Toy computation:** process tokens one at a time with `STEP`, `AUTO RUN`, and `RESET`.
- **State inspector:** inspect the accumulated matrix and query readout.
- **Prediction challenge:** compare a learner's estimate with the computed result.
- **BDH/BDH-CQ section:** research-grounded context, explicitly separated from the toy implementation.
- **60-second challenge:** a concise synthesis check.

---

## 5. Evidence and Substrate Labels

The site deliberately distinguishes different evidence levels:

| Label | Meaning |
|---|---|
| **[RESEARCH EVIDENCE]** | Claim or result grounded in a cited primary research source. |
| **[FOUNDATIONAL / BACKGROUND]** | Earlier work used to establish the conceptual lineage. |
| **[BDH TECHNICAL REPORT]** | Architectural material reported by Pathway's BDH/BDH-CQ research. |
| **[CONCEPTUAL RELATION]** | A connection or analogy that is useful pedagogically but is not presented as equivalence. |
| **[TOY COMPUTATION]** | Deterministic browser-side computation implemented specifically for teaching. |

### Live vs. static components

| Component | Status |
|---|---|
| Sequence-length scaling | **Live computation** |
| Canvas interaction heatmap | **Live rendering** |
| 4D state update | **Live toy computation** |
| Query probe | **Live toy computation** |
| Hero pipeline | **Live deterministic toy computation + animation** |
| BDH diagrams/text | **Research-grounded explanatory content** |
| Literature | **Static citations / references** |

---

## 6. Technical Mechanism

### 6.1 Standard attention

For the usual softmax attention formulation:

`Attention(Q,K,V) = softmax(QK^T / √d) V`

The attention score structure contains `N × N` query-key entries for a sequence of length `N`.

This project uses the `N²` count in its scaling visualization as a **simplified educational proxy for the pairwise score-matrix size**. It should not be interpreted as a universal statement about wall-clock runtime or about every optimized attention implementation.

### 6.2 Kernelized / linear-attention view

A feature map `φ(·)` can be used so that the similarity is represented through feature-space inner products such as:

`sim(q,k) = φ(q)^T φ(k)`

This allows a causal formulation to regroup the computation so that the key/value contribution can be accumulated first. One useful recurrent form is:

`S_t = S_(t-1) + φ(k_t) v_t^T`

with a corresponding normalizer accumulator such as:

`z_t = z_(t-1) + φ(k_t)`.

A normalized readout can then be expressed in the form:

`y_t = [φ(q_t)^T S_t] / [φ(q_t)^T z_t]`.

The exact feature map, normalization, gating, and implementation details vary across linear-attention methods; Memory Lab intentionally visualizes one small, transparent formulation rather than claiming universality.

---

## 7. Why the 100-Token Race Uses `N²` vs `N`

The scaling experiment visualizes two different computational organizations:

### Pairwise attention view

The score matrix has `N × N` entries.

At `N = 100`:

`100 × 100 = 10,000` score positions.

### Incremental state view

The recurrent formulation can update its contextual state once per incoming token.

At `N = 100`:

`100` sequential state updates.

This is a **teaching visualization of sequence-length dependence**, not a benchmark of real GPU latency.

It also intentionally separates two different ideas that are often conflated:

- the `N²` structure of a full attention score matrix, and
- the `O(N)` storage growth of an autoregressive KV cache.

---

## 8. Toy Computation Implemented in the Browser

The core experiment uses a small feature dimension (`d = 4`) so that every state change can remain visible.

For each token, the browser computes:

1. `φ(k_t)` — deterministic feature representation
2. `ΔS_t = φ(k_t) v_t^T` — rank-1 outer-product write
3. `S_t = S_(t-1) + ΔS_t` — cumulative state
4. `z_t = z_(t-1) + φ(k_t)` — normalizer accumulator
5. query/readout from the current state

The values are deterministic and are not presented as model benchmark results.

### Important scope

This toy system is **not**:

- an official BDH implementation;
- a production Linear Attention implementation;
- a benchmark of model quality;
- evidence that any particular architecture is always faster or more accurate.

---

## 9. BDH / BDH-CQ Connection

Pathway's documentation describes **BDH** as a brain-inspired Post-Transformer architecture family and describes a connection between its memory mechanisms and attention/synaptic formulations. The Pathway brief also notes that BDH-GPU is a separate GPU-friendly formulation using ReLU/low-rank transformations with linear attention and explicitly cautions against classifying BDH as a Mamba-style SSM.

For **BDH-CQ**, Pathway describes contextual memory in relation to attention, fast-weight memory, and linear-attention views of contextual association, including additive accumulation of contextual state in a special case.

Memory Lab therefore presents this relationship as:

`Linear Attention → incremental associative state → contextual-memory perspective → BDH / BDH-CQ research connection`

### What we do not claim

- `BDH = Linear Attention`
- `BDH = Mamba / SSM`
- `Toy state matrix = official BDH state`
- `Toy animation = a reproduction of BDH model behavior`

The BDH portion is included to explain a **research connection**, not to imply that the toy demo reproduces the full architecture.

---

## 10. Limitations

### 10.1 Finite associative capacity

A fixed-size state cannot be assumed to preserve arbitrary histories perfectly. As the amount of information grows, interference and retrieval degradation can become important.

### 10.2 Approximation and retrieval trade-offs

Replacing the standard softmax similarity structure with a feature-map formulation changes the retrieval behavior and may reduce the ability to form extremely sharp token-specific attention distributions.

### 10.3 Theory is not wall-clock performance

Asymptotic sequence scaling does not by itself determine real hardware latency. Kernel fusion, memory traffic, tiling, dimensions, sequence length, and implementation quality all matter.

### 10.4 Educational simplification

The browser implementation uses a tiny 4D toy state for transparency. Modern models use much larger representations, multiple layers/heads or alternative architectural mechanisms, and additional optimizations not represented here.

---

## 11. Recent Primary Research

The submission uses recent primary research to situate the concept, while older papers are retained as foundations.

### Recent primary research — 2022–2024

1. **Qin et al. (2022), _CosFormer: Rethinking Softmax in Attention_** — ICLR 2022. A linear-attention approach that uses cosine-based reweighting to improve the behavior of linearized attention.  
   https://arxiv.org/abs/2202.08791

2. **Sun et al. (2023), _Retentive Network: A Successor to Transformer for Large Language Models_** — introduces a retention/recurrent formulation supporting parallel and recurrent computation and efficient inference.  
   https://arxiv.org/abs/2307.08621

3. **Yang et al. (2024), _Gated Linear Attention Transformers with Hardware-Efficient Training_** — studies gated linear attention and hardware-efficient training, including practical implementation considerations beyond asymptotic complexity.  
   https://arxiv.org/abs/2312.06635

### Foundational / background

4. **Katharopoulos et al. (2020), _Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention_**.  
   https://arxiv.org/abs/2006.16236

5. **Schlag et al. (2021), _Linear Transformers Are Secretly Fast Weight Programmers_**.  
   https://arxiv.org/abs/2102.11174

### BDH research

6. **Pathway Research, Dragon Hatchling (BDH) / BDH-CQ technical reports and related research materials.** Use the primary Pathway sources supplied with the hackathon materials for architecture-specific claims.

---

## 12. Reproducibility

No backend or package manager is required.

### Run locally

```bash
git clone https://github.com/<your-username>/linear-attention-explainer.git
cd linear-attention-explainer
python -m http.server 8000
```

Open:

`http://localhost:8000`

The project can also be opened directly through `index.html` in a modern browser.

### Reproduce the main claim

1. Open the **100-Token Race**.
2. Set `N = 16`, then `32`, `64`, and `100`.
3. Observe the pairwise score-matrix size grow as `N²`.
4. Observe the incremental state-update stream grow once per token.
5. Open the core experiment and use **STEP NEXT** to inspect the state update token by token.
6. Use **PREDICT → RUN** to test the scaling relationship before seeing the result.

---

## 13. Architecture

```text
index.html
    │
    ├── 01 Hero
    ├── 02 Pairwise Footprint
    ├── 03 100-Token Race
    ├── 04 Core Memory Experiment
    ├── 05 Prediction
    ├── 06 BDH / BDH-CQ Connection
    ├── 07 Limitations
    └── 08 Literature / Sources

style.css
    └── dark research-lab visual system, responsive layout, animation and accessibility states

script.js
    ├── deterministic toy linear-algebra engine
    ├── sequence scaling controls
    ├── Canvas heatmap renderer
    ├── hero state machine
    ├── experiment controls
    ├── query probe
    ├── quiz / challenge logic
    └── scroll progress / section state
```

---

## 14. AI Assistance Disclosure

Generative AI tools were used during development for:

- interaction and information-architecture ideation;
- frontend code generation and refactoring;
- debugging and browser-test assistance;
- documentation drafting;
- research summarization and source organization.

The project team is responsible for the final implementation and must be able to explain and defend the code, equations, visual mappings, research claims, and citations used in the submission. This distinction between AI assistance and technical ownership is intentional.

---

## 15. Assets, Dependencies, and Licenses

### Code

The project code is intended for academic/educational reuse. The repository may use an **MIT License** if that is the license selected by the team for the final repository.

### Fonts

If externally hosted fonts are retained, keep their original license notices and source links in the repository.

### Formula rendering

If KaTeX or another external formula renderer is retained, document its version, source, and license in the repository.

### Graphics

The primary visualizations are generated with HTML/CSS/SVG/Canvas rather than copied from third-party images.

---

## 16. Scientific Honesty Checklist

Before submission, verify that:

- [ ] The main claim is stated in one sentence.
- [ ] The `N²` display is described as an attention-score-matrix / educational proxy, not a universal runtime equation.
- [ ] KV-cache storage is described as growing linearly with sequence length.
- [ ] The toy state is not presented as an official production model.
- [ ] BDH is not presented as identical to Linear Attention.
- [ ] BDH is not classified as a Mamba-style SSM.
- [ ] At least three recent primary papers from 2022–2026 are cited.
- [ ] Research claims are checked against the primary sources.
- [ ] Live, toy, conceptual, and research content are clearly distinguished.
- [ ] AI assistance and assets/licenses are disclosed.

---

## 17. Submission Context

This project is designed for the **DataForge 2026 Pathway Track — Explain the Frontier**. The Pathway brief asks for a single clear concept, an interactive substrate that lets the learner change something meaningful and observe the concept, a substantive BDH/BDH-CQ connection, recent primary sources, a public repository, a complete README, reproducibility information, source/license disclosure, AI disclosure, and a one-page concept summary. The project is therefore intentionally built around one claim and a guided `observe → interact → explain` learning journey.

---

## 18. Team Contribution

This project was developed as a student-built educational artifact. Team members should list their actual contributions in the final repository or submission form (e.g. frontend implementation, research verification, interaction design, testing, documentation).

---

**MEMORY LAB**  
*Understanding Linear Attention through an interactive memory state.*
