<h1 align="center">Text-JEPA</h1>

**Sofia Bodnar**

[\[LeCun JEPA\]](https://openreview.net/pdf?id=BZ5a1r-kVsf)
[\[I-JEPA\]](https://arxiv.org/abs/2301.08243)
[\[V-JEPA\]](https://arxiv.org/abs/2404.08471)
[\[LLM-JEPA\]](https://arxiv.org/abs/2509.14252)
[\[LeJEPA\]](https://arxiv.org/abs/2511.08544)

---

## Hypothesis

Language is not the content of thought; it is a coordinate system for navigating it. A sentence does not carry an idea; it is an address to one. "The brown cat chased the green painted rat" has never existed in your mind before, yet the sentence deposits it there by composing known coordinates into a new location in a shared latent space.

Recent work shows that all major open-weight language models independently converge on the same structure: early layers encode surface language, middle layers operate in a language-agnostic concept space, and final layers translate back to surface form. The thinking happens in the latent space. Language is the interface.

Current language models do not exploit this. They are good at retrieval — accumulating knowledge and regurgitating it at the right time — but not understanding. After sitting through ten meetings, a person builds a model of the company: they can predict how someone will behave, anticipate what a project will produce, explain behavior that looks irrational from a partial view. That is a world model, a latent understanding of a system's dynamics built from observing its trajectories. Not pattern matching over tokens.

Text-JEPA is an attempt to build this for language. Rather than predicting the next token, it predicts the next address: given the current trajectory through idea space, where does it go next? The claim is that a model trained this way learns something closer to understanding than retrieval — a structured, non-collapsed idea space whose dynamics transfer to reasoning tasks the model was never trained on.

---

## Why JEPA for text

JEPA was designed for sensory data — images and video contain large amounts of unpredictable low-level noise (pixel values, lighting, texture) that carry no semantic signal. JEPA filters this out by predicting in latent space rather than pixel space.

Language is different. It is already symbolic and compressed. Word choice is already high-level. Autoregressive prediction works well precisely because tokens are already semantic units.

But the noise in language is not sensory — it is surface form. "He ran away" and "he fled" address the same location in idea space via different surface paths. Autoregressive training treats these as different targets. Text-JEPA treats them as the same: the prediction target is the idea-space address, not the token sequence.

The problem autoregressive training solves: given this prefix, what words come next?
The problem Text-JEPA solves: given this trajectory through idea space, where does it go next?

---

## Method

k context sentences are encoded by the context encoder (trained jointly) into idea state embeddings. In parallel, sentence k+1 is encoded by the target encoder (EMA copy, stop-gradient) into a target embedding. The predictor — a 2-layer bidirectional transformer — takes the k context embeddings plus a learned mask token and outputs a prediction at position k+1. Loss is L2 between predicted and target embeddings on the unit hypersphere, plus SigReg (BCS) which enforces an isotropic Gaussian distribution over embeddings to prevent collapse.

The predictor has no access to raw tokens. It operates entirely in learned latent space. At inference, the target encoder is not used — the predictor rolls forward in idea space directly.

---

## Architecture

**Context encoder.** OPT-125m backbone, trained jointly with the predictor under the JEPA objective. Mean pool last hidden states → linear projection → 256d. Must be trained jointly: a frozen encoder learns retrieval features, not temporal dynamics.

**Target encoder.** EMA copy of the context encoder. Weights never receive gradients directly. Momentum cosine-scheduled from 0.996 → 1.0. The EMA asymmetry is what makes JEPA non-degenerate: without it, both encoders collapse to a trivial solution. The target encoder is always slightly behind, creating a moving target that cannot be matched by collapse.

**Predictor.** 2-layer bidirectional transformer, 4 heads, mlp_ratio=2. Input: k=4 context embeddings + learned mask token with learned positional embeddings. Output: 256d prediction at mask position. Intentionally lightweight — representation quality lives in the encoder, not the predictor.

**No decoder.** The predictor outputs a 256d vector in idea space. There is no module mapping predictions back to text. Text production is downstream; thinking happens in latent space first.

---

## Geometry

An address is a position, not a direction. The geometry of idea space is the unit hypersphere.

All embeddings are normalized to the sphere before loss. L2 on the sphere is positional — the predictor is correct only if it arrives at the right address, not merely the right direction. Ray collapse (all sentences mapping to vectors pointing in the same direction at different magnitudes) is geometrically impossible with fixed magnitude.

If geometry is poorly structured, rollout predictions drift and stop corresponding to reality. Geometry is not just a theoretical choice — it is what makes long-horizon prediction possible.

Collapse prevention uses two complementary mechanisms:
- **EMA** prevents dynamic collapse: decouples the two encoder optimization trajectories
- **SigReg (BCS)** prevents geometric collapse: enforces isotropic Gaussian distribution over embeddings, ensuring distinct sentences occupy distinct addresses

Full objective: `L = L2(normalize(z_pred), sg(normalize(z_target))) + λ · SigReg`

---

## Training

| Parameter | Value |
|---|---|
| Dataset | ROCStories (~78k sentence windows) |
| Unit | sentence |
| Context window | k=4, predict sentence k+1 |
| Augmentation | token dropout p=0.15 on context inputs only; target always clean |
| Backbone | OPT-125m |
| Latent dim | 256 |
| Optimizer | AdamW, lr=3e-4, cosine decay to 1e-5 |
| Batch size | 16 |
| Max epochs | 200, early stopping patience 30 |
| EMA momentum | cosine 0.996 → 1.0 |
| SigReg λ | 1.0 |
| Seeds | 5 |
| Split | 70/10/20 temporal, no future leakage |

---

## Evaluation

| Eval | Dataset | Metric | Pass condition |
|---|---|---|---|
| 1 — Predictor vs copy-forward | ROCStories | L2 on unit sphere | model < copy-forward, p < 0.05 |
| 3 — Long-horizon rollout | ROCStories | cosine at steps +1, +2, +3 | > 0.5 at step +3 |
| 4 — Linear probe | ARC-Easy, ARC-Challenge, GSM8K | accuracy, frozen encoder | above random baseline |
| 5 — Effective rank | ROCStories | exp(normalized eigenvalue entropy) | erank > 76.8 (0.3 × 256) |

Primary claim: eval 1. Surprising result if it happens: eval 4 shows JEPA pretraining improves reasoning accuracy over raw OPT-125m zero-shot without any reasoning supervision.

---

## Extension: temporal graph-JEPA

Text-JEPA is the proof of concept on linear narrative sequences. The natural extension asks: can you learn a representation of a person's organizational state that is predictable from the communication graph around them?

The unit becomes a person's position in an organizational network — who they communicate with, at what volume, how that local structure shifts over time. The context is the communication subgraph around a person over k weeks. The target is that person's state the following week.

Training ground: Enron corpus (500k emails, 3 years, known organizational events that fractured communication structure at specific points). Same JEPA principle, relational signal instead of sequential.

---

## Getting started

```bash
pip install -r requirements.txt
```

**Training (Modal, GPU):**
```bash
modal run experiments/modal_train.py
```

**Evaluation:**
```bash
python experiments/eval_all.py
```

## Requirements

```
torch >= 2.0
transformers >= 4.40
datasets
numpy
scikit-learn
scipy
modal
pytest
```
