---
title: Agent Context — Text-JEPA
type: context
date: 2026-04-22
tags: [context, spec, invariants, protocol]
---

# Agent Context — Text-JEPA

Read this file first. Every session. Before writing a single line of code.

---

## Commit Protocol

**Never commit.** When a phase is complete and the codebase is in a clean, tested state, output exactly:

```
READY TO COMMIT: [one line describing what changed]
FILES: [list of files changed]
```

Then stop. Do not run `git add` or `git commit`. Sofia commits manually.

---

## Iteration Protocol

This is not a one-shot build. Work in phases. After each phase, stop and report. Do not proceed to the next phase without explicit instruction. Each phase produces a concrete artifact before the next begins.

Phase gates:
- Phase 1 complete: plan written to `vault/plans/`, reviewed
- Phase 2 complete: `pytest tests/ -v` passes, all tests green, no model code yet
- Phase 3 complete: training runs on Modal for at least 1 seed, loss curves logged
- Phase 4 complete: all 6 evals run across 5 seeds, results in vault
- Phase 5 complete: write-up in vault, honest negatives documented

At each gate, output `PHASE [N] COMPLETE:` followed by a summary and what to do next. Do not auto-proceed.

---

## Research History — Read This to Understand the Hypothesis

Before implementing anything, read the following files. First-principles understanding of why we are building this prevents architectural drift.

**Full vault locations to read (all transcripts and context):**
- `jepa-sandbox/vault/transcript/` — read ALL files 01 through 09 in order
- `jepa-sandbox/vault/` — read all .md files (research-brief, text-jepa-v2-spec, frozen-config, graph-jepa-context, tgjepa-session-context)
- `eb_jepa/vault/transcript/lecun-jepa-world-models-talk.md` — LeCun talk transcript
- `~/ObsidianVaults/claude-memory/` — read MEMORY.md index, then all referenced files

**Primary research context (jepa-sandbox repo):**
- `jepa-sandbox/vault/transcript/06-poc-v1-analysis.md` — the text JEPA PoC failure in detail: what ran, what broke, why frozen BGE failed, what it taught us (this is the direct predecessor to this project)
- `jepa-sandbox/vault/transcript/04-sofia-text-jepa-poc-notes.md` — original PoC design decisions and open questions
- `jepa-sandbox/vault/transcript/05-sofia-text-jepa-conceptual-framing.md` — the latent subspace argument: why text JEPA is about idea space, not token space
- `jepa-sandbox/vault/transcript/02-text-jepa-framing-and-toy-experiment.md` — Ashwin's framing: the ball-bouncing analogy, why latent space = understanding
- `jepa-sandbox/vault/transcript/03-jepa-world-models-and-org-prediction.md` — the distinction between simulation and latent understanding
- `jepa-sandbox/vault/transcript/07-lecun-jepa-world-models-talk.md` — LeCun on trained encoders, why frozen encoders fail, SigReg, information maximization
- `jepa-sandbox/vault/transcript/09-jepa-explainer-lega-sigreg.md` — collapse prevention lineage, LeGa/SigReg, why JEPA works on structured data
- `jepa-sandbox/vault/research-brief.md` — JEPA foundations, loss functions, prior work survey
- `jepa-sandbox/vault/text-jepa-thinking-space.tex` (or the compiled PDF) — the full proposal for this project: architecture, adversarial interpreter, two-step prediction, eval protocol

**Why the hypothesis is what it is:**
The text JEPA PoC (Harry Potter, frozen BGE, 199 chapters) passed 5 evals with a 6K-param linear extrapolator by 0.6% margin over copy-forward. This is not a result — it is a failure that teaches something. The failure mode: frozen BGE encodes retrieval features (semantic similarity), not temporal dynamics. Consecutive chapters cluster tightly in BGE space. There is almost no directional signal because the encoder was trained to find similar things, not to extract what changes over time. LeCun's information maximization argument explains why this always happens with frozen encoders. Text-JEPA fixes this by training the encoder jointly with the predictor under the JEPA objective, forcing it to learn only what is predictable from context.

The hypothesis (predicting in idea space beats copy-forward) is not obvious. It failed once already with a frozen encoder. The question is whether a jointly trained encoder on richer, more structured data (PG-19, ROCStories) can learn the dynamics that frozen BGE could not.

---

## What We Are Building

A JEPA whose input is text. Not next-token prediction. Not a language model. A system that predicts in a learned latent idea space, where text is the interface and the latent space is the medium.

The core insight: language is a compression scheme for a space of ideas. A sentence is a sequence of operations on the reader's concept space — addressing a concept, modifying it, composing it with others. Text-JEPA learns the encoder that maps text to that space and the predictor that models its dynamics.

This is distinct from LLM-JEPA (Huang, LeCun, Balestriero 2025), which applies the JEPA objective to LLM pretraining. In Text-JEPA, the latent space is the primary medium. Text production is a downstream decoder step, not the objective.

**Lineage:** I-JEPA (images) → V-JEPA (video) → LLM-JEPA (text as LLM objective) → **Text-JEPA (text, latent idea space as primary medium)**

---

## Research Question

> Can JEPA be applied to text by treating language as navigation of a latent idea space, and does predicting in that space learn representations encoding semantic dynamics better than next-token prediction?

---

## Hypothesis

**Primary:** A model trained to predict the next idea state (trained encoder, latent predictor) outperforms a copy-forward baseline on sequential text comprehension tasks (ROCStories, HellaSwag). The encoder learns to extract dynamically predictable features rather than retrieval-relevant ones.

**Secondary:** Learned representations improve downstream reasoning task performance (ARC, GSM8K) over a frozen retrieval encoder baseline.

**Negative result condition:** If the predictor cannot beat copy-forward on structured corpora (PG-19, ROCStories), this is evidence that idea-space dynamics are not recoverable from surface text alone. This motivates adding relational structure (graph inputs) as a complementary signal. It is a valid finding. Report it.

---

## Architecture — Invariants (Red Flagger Checks These)

These must hold at every point in the codebase. If any are violated, stop and fix.

1. **No token reconstruction.** The loss is never cross-entropy on tokens. The only prediction target is a latent embedding. If a token-level loss appears anywhere, it is wrong.
2. **Trained encoder.** The text encoder trains jointly with the predictor. No frozen embeddings in the JEPA training path. A frozen BGE or nomic-embed was trained for retrieval, not temporal dynamics.
3. **EMA target encoder.** The target encoder is a slow-moving EMA copy of the context encoder. It does not receive gradients.
4. **Stop-gradient on target encoder outputs.** Every tensor from the target encoder has sg() applied before it enters any computation graph.
5. **SigReg from eb_jepa.** Use the BCS class from `eb_jepa/losses.py` as-is. Do not reimplement. Do not substitute VICReg unless SigReg is demonstrably unstable (document this if it happens).
6. **Predictor operates in latent space only.** The predictor takes encoded idea states as input. It never sees raw tokens.
7. **Masking at predictor stage only.** The context encoder processes full input segments. Masking is applied only when constructing the predictor input: the target segment embedding is replaced by a learnable mask token.
8. **Decoder is decoupled.** The decoder (latent -> text) is trained separately and is not part of the JEPA objective. The predictor is not penalized for decoder output quality.

---

## Architecture — Spec

### Context Encoder (Trained Jointly)
- Input: tokenized text segment (sentence, paragraph, or document chunk)
- Output: d=256 dimensional latent vector (idea state)
- Architecture: transformer encoder with learned projection head
- Initialization: small pretrained LM (LLaMA-3.2-1B or Gemma-2B), fine-tuned end-to-end
- Parameter count: document after implementation

### Target Encoder (EMA, Stop-Grad)
- Exact copy of context encoder
- EMA update: theta_target = m * theta_target + (1 - m) * theta_context
- Momentum m: cosine-scheduled 0.996 → 1.0
- All outputs: stop-gradient applied

### Predictor
- Input: k encoded idea states z[t-k:t] from context encoder + mask token at t+1
- Architecture: 2-layer bidirectional transformer, 4 heads, dim 256, MLP ratio 2
- Positional encoding: learnable temporal position embeddings
- Output: predicted embedding z_pred at mask position
- Parameter count: ~500K-1M (document after implementation)

### Loss
```
L = SmoothL1(z_pred, sg(z_target)) + lambda_reg * L_SigReg
```
SmoothL1 preferred over L2 for robustness to outlier segments with large embedding shifts (following V-JEPA motivation). lambda_reg frozen before any eval run.

### Decoder (Optional, Decoupled)
- Maps latent idea state back to text for interpretability and qualitative eval
- Architecture: small autoregressive LM head conditioned on latent vector
- Not part of JEPA training objective
- Trained separately after JEPA training is complete

---

## Brownfield Map

```
REFERENCE:    ijepa/              → I-JEPA training loop and JEPA machinery
              jepa/               → V-JEPA temporal predictor design
              lm-evaluation-harness/ → downstream eval harness (HellaSwag, ARC, GSM8K)
DO NOT MODIFY: ijepa/, jepa/, lm-evaluation-harness/
BUILD FROM SCRATCH: src/ (encoder, predictor, loss, train, eval)
```

Before writing any src/ code, read `ijepa/src/` and `jepa/src/` to understand how existing JEPA implementations structure their training loops, mask collators, and EMA updates. Match these patterns where possible.

---

## Datasets

| Dataset | How to load | Notes |
|---|---|---|
| PG-19 | `datasets.load_dataset("pg19")` | 10K+ books, use HuggingFace datasets |
| ROCStories | CSV download from official site | 50K 5-sentence stories |
| HellaSwag | `lm-evaluation-harness/` | 70K commonsense completion examples |
| ARC | `lm-evaluation-harness/` | 7.8K science reasoning questions |
| GSM8K | `lm-evaluation-harness/` | 8.5K math reasoning chains |

Segment granularity: sentence-level for ROCStories (each sentence is one idea state), paragraph-level for PG-19. Context window k=4 segments. Prediction target: segment t+1.

---

## Evaluation Protocol

5 random seeds. Paired Wilcoxon signed-rank test for all pairwise comparisons. Bonferroni correction across the full evaluation family. Report mean +/- 95% CI. No metric added or removed after results observed. Single-seed numbers are diagnostic only.

| Eval | What it tests | Pass criterion |
|---|---|---|
| 1 — Prediction accuracy | Does predictor beat copy-forward? | model > copy-forward, p < 0.05 after Bonferroni |
| 2 — HellaSwag ranking | Does predicted embedding rank correct continuation above 3 distractors? | accuracy > copy-forward ranking baseline |
| 3 — Long-horizon rollout | Does prediction degrade gracefully at k=1,2,4 steps? | stays above copy-forward at k >= 2 |
| 4 — Downstream reasoning | Do representations improve ARC and GSM8K probes? | beats frozen encoder on >= 2/4 tasks, p < 0.05 |
| 5 — Representation quality | Are representations non-collapsed? | effective rank > 0.3 * d |
| 6 — Calibration | Are confidence estimates calibrated? (adversarial variant only) | ECE < 0.1 |

Primary metric: Eval 1 (prediction accuracy vs copy-forward).

---

## Training Protocol

Temporal split. No future leakage.
- Train: first 70% of corpus (by sequence position)
- Val: next 10%
- Test: final 20%

Hyperparameters frozen before any eval run. Document in `docs/frozen-config-textjepa.md`.

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| LR schedule | cosine anneal to 1e-5 |
| Batch size | 16 |
| Max epochs | 200 |
| Early stopping | patience 30 on val loss |
| Context window k | 4 segments |
| EMA momentum start | 0.996 |
| EMA momentum end | 1.0 |

---

## Workflow Protocol

**Phase 1 — Planning.** Read this file. Read README. Read `ijepa/` and `jepa/` reference repos. Write implementation plan to `vault/plans/`. Do not write model code yet.

**Phase 2 — TDD.** Write all tests before any model code. Tests run on CPU, synthetic text inputs, under 30 seconds total. `pytest tests/ -v` must pass before phase 3.

Test checklist:
- context encoder: correct output shapes, trainable params
- target encoder: requires_grad=False, EMA updates correctly, diverges from source
- stop-gradient: verified on target encoder outputs during backprop
- predictor: bidirectional attention, full context changes mask output
- mask token: present in predictor input, learnable
- SmoothL1 loss: decreases when prediction approaches target
- SigReg: detects collapsed representations (all identical embeddings)
- no token reconstruction: no cross-entropy loss anywhere in training path
- one full training step: context encoder params change, target encoder changes via EMA only
- temporal split: no future segment leakage

**Phase 3 — Execution.** Implement per spec. Modal for GPU training.

**Phase 4 — Review.** Red flagger checks all 8 invariants. Full eval suite. Write results to vault.

**Phase 5 — Vault write-up.** Findings as Obsidian notes in `vault/results/`. Honest negatives get their own note.

---

## Subagent Responsibilities

| Agent | Scope |
|---|---|
| data-agent | Dataset loaders for all 5 datasets, segment extraction, temporal splits |
| model-agent | Context encoder, target encoder, predictor, EMA logic, decoder |
| test-agent | All test files, synthetic inputs only, pytest passing |
| training-agent | Modal training scripts, config files, seed management |
| eval-agent | All 6 evals, Wilcoxon tests, Bonferroni, CI, lm-evaluation-harness integration |
| red-flagger | Checks 8 invariants + hypothesis validity every 10 minutes, writes to vault/redflag-reports/ |

---

## Red Flagger Invariants

Output format: PASS / WARN / FAIL per line. Any FAIL stops all implementation.

```
1. no token reconstruction? (no cross-entropy or token-level loss in training path)
2. encoder trained jointly? (no frozen encoder in training path)
3. target encoder EMA only? (no optimizer touches target encoder params)
4. stop-gradient on target encoder outputs? (sg() applied before loss)
5. sigreg from eb_jepa? (BCS class used, not reimplemented)
6. predictor input is latent only? (no raw tokens fed to predictor)
7. masking at predictor stage only? (context encoder sees full segment)
8. decoder decoupled? (no decoder loss in JEPA training objective)
HYPOTHESIS: is eval 1 actually measuring latent prediction vs copy-forward?
LEAKAGE: does any training example see future segment content?
COLLAPSE: is effective rank of embeddings monitored during training?
```

---

## Vault Writing Style

Write all findings, decisions, and results as Obsidian markdown notes under `vault/`.

```
vault/
    plans/              implementation plans before coding
    results/            eval results, one note per experiment
    decisions/          architectural decisions and their reasoning
    redflag-reports/    red flagger output, one file per run
```

Frontmatter format:
```yaml
---
title: [descriptive title]
type: result | decision | plan | redflag
date: YYYY-MM-DD
tags: [relevant, tags]
related: [[other-note]]
---
```

One note per experiment or decision. Link with [[double brackets]]. Honest negatives get their own note. If something failed, write why. Future sessions depend on this record being accurate.
