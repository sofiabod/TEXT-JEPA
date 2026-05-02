<h1 align="center">Text-JEPA</h1>

**Sofia Bodnar**

[\[LeCun JEPA\]](https://openreview.net/pdf?id=BZ5a1r-kVsf)
[\[I-JEPA\]](https://arxiv.org/abs/2301.08243)
[\[V-JEPA\]](https://arxiv.org/abs/2404.08471)
[\[LLM-JEPA\]](https://arxiv.org/abs/2509.14252)
[\[LeJEPA\]](https://arxiv.org/abs/2511.08544)

---

## Hypothesis

Language is not the content of thought; it is a coordinate system for navigating it. A sentence does not carry an idea; it is an address to one.

Recent work shows all major open-weight LLMs converge on the same internal structure: early layers encode surface language, middle layers operate in a language-agnostic concept space, final layers translate back. The thinking happens in latent space. Language is the interface.

Current models do not exploit this. They are good at retrieval, not understanding. After ten meetings, a person builds a model of the company: predicting behavior, anticipating outcomes, explaining what looks irrational from a partial view. That is a world model built from observing trajectories, not pattern matching over tokens.

Text-JEPA predicts the next address in idea space rather than the next token. The claim is that a model trained this way learns something closer to understanding: a structured, non-collapsed idea space whose dynamics transfer to reasoning tasks it was never trained on.

---

## Why JEPA for text

JEPA filters unpredictable low-level noise by predicting in latent space. For images, that noise is pixels. For text, it is surface form: "he ran away" and "he fled" address the same location via different surface paths. Autoregressive training treats them as different targets. Text-JEPA treats them as the same.

---

## Method

k context sentences are encoded by the context encoder into idea-state embeddings. Sentence k+1 is encoded by the target encoder (EMA copy, stop-gradient) into a target embedding. A 2-layer predictor takes the k context embeddings plus a learned mask token and outputs a prediction at position k+1. Loss is L2 on the unit hypersphere plus SigReg (BCS) to prevent collapse.

At inference the target encoder is not used. The predictor rolls forward in idea space directly.

---

## Architecture

**Context encoder.** OPT-125m, trained jointly with the predictor. Mean pool last hidden states, linear projection to 256d. Must train jointly: a frozen encoder learns retrieval features, not dynamics.

**Target encoder.** EMA copy of the context encoder, no gradients. Momentum cosine-scheduled 0.996 to 1.0. The asymmetry prevents collapse: the target is always slightly behind, making a trivial zero solution impossible.

**Predictor.** 2-layer bidirectional transformer, 4 heads. Intentionally lightweight: representation quality lives in the encoder.

**No decoder.** The predictor outputs a 256d vector in idea space. Nothing maps predictions back to tokens.

---

## Geometry

All embeddings are normalized to the unit hypersphere before loss. L2 on the sphere is positional: the predictor is correct only at the right address, not merely the right direction.

Collapse prevention:
- **EMA** decouples the two encoder optimization trajectories
- **SigReg (BCS)** enforces an isotropic Gaussian over embeddings so distinct sentences occupy distinct addresses

`L = L2(normalize(z_pred), sg(normalize(z_target))) + λ · SigReg`

---

## Training

| Parameter | Value |
|---|---|
| Dataset | ROCStories (~78k sentence windows) |
| Context window | k=4, predict sentence k+1 |
| Augmentation | token dropout p=0.15 on context; target always clean |
| Backbone | OPT-125m |
| Latent dim | 256 |
| Optimizer | AdamW, lr=3e-4 (head), 1e-5 (backbone), cosine to 1e-5 |
| Batch size | 16 |
| Early stopping | patience 30 |
| EMA momentum | cosine 0.996 to 1.0 |
| Seeds | 5 |
| Split | 70/10/20 temporal |

---

## Evaluation

| Eval | Dataset | Metric | Pass condition |
|---|---|---|---|
| 1 | ROCStories | L2 vs copy-forward baseline | model < baseline, p < 0.05 |
| 3 | ROCStories | cosine at rollout steps +1, +2, +3 | > 0.5 at step +3 |
| 4 | ARC, GSM8K | linear probe accuracy, frozen encoder | above random |
| 5 | ROCStories | effective rank | erank > 76.8 (0.3 x 256) |

Eval 4 is the result that would matter: JEPA pretraining improves reasoning accuracy with no reasoning supervision.

---

## Extension

Text-JEPA is the proof of concept on linear narrative sequences. The extension: can you learn a representation of a person's organizational state that is predictable from the communication graph around them? The unit becomes a person's position in an organizational network over k weeks. Training ground: Enron corpus. Same JEPA principle, relational signal instead of sequential.

---

## Getting started

```bash
pip install -r requirements.txt
modal run experiments/modal_train.py
modal run experiments/modal_eval.py
```
