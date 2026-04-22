<h1 align="center">Text-JEPA</h1>

PyTorch codebase for **Text-JEPA** (Text-based Joint-Embedding Predictive Architecture), a method for self-supervised learning of latent idea-space representations from text.

[\[LeCun JEPA\]](https://openreview.net/pdf?id=BZ5a1r-kVsf)
[\[I-JEPA\]](https://arxiv.org/abs/2301.08243)
[\[V-JEPA\]](https://arxiv.org/abs/2404.08471)
[\[LLM-JEPA\]](https://arxiv.org/abs/2509.14252)
[\[LeJEPA\]](https://arxiv.org/abs/2511.08544)

## Method

Text-JEPA is a method for self-supervised learning from text. Rather than predicting the next token, Text-JEPA predicts the latent representation of the next text segment given the representations of prior segments. The key insight is that language is a compression scheme for a latent idea space: a sentence does not transmit pixels or tokens but acts as a sequence of operations on the reader's concept space, addressing, modifying, and composing ideas. Text-JEPA learns the encoder that maps text into that space and the predictor that models its dynamics.

At a high level, Text-JEPA predicts representations of future text segments from representations of earlier ones. Notably, this approach:
1. Does not rely on token-level reconstruction or next-token cross-entropy loss.
2. Trains the encoder jointly with the predictor so that it learns to extract dynamically predictable features, not retrieval-relevant ones.
3. Uses SigReg (LeJEPA, 2025) to prevent representation collapse without requiring contrastive negatives or EMA alone.

```
text segments [t-k, ..., t]
        |
        v
  context encoder (trained jointly)        target encoder (EMA, stop-grad)
  transformer + projection head            slow-moving copy of context encoder
  input: text segments t-k to t            input: text segment t+1
  output: idea state embeddings z[t-k:t]   output: target embedding z_target[t+1]
        |                                           |
        v                                           |
  predictor                                         |
  2-layer bidirectional transformer                 |
  input: z[t-k:t] + mask token at t+1              |
  output: z_pred[t+1]  --------------------------> loss
        |
        v
  SigReg (LeJEPA / BCS loss)
  forces embedding distribution toward isotropic Gaussian
  prevents representation collapse
```

The predictor has no access to tokens at prediction time. It operates entirely in the learned latent space.

## Approach

**Context encoder.** A transformer encoder with a learned projection head, initialized from a small pretrained LM (LLaMA-3.2-1B or Gemma-2B) and fine-tuned end-to-end under the JEPA objective. A frozen encoder trained for retrieval learns semantic similarity features, not temporal dynamics. Only a jointly trained encoder learns to extract what is predictable.

**Target encoder.** Exact architectural copy of the context encoder. Updated via EMA of the context encoder weights (momentum cosine-scheduled 0.996 -> 1.0). All outputs are stop-gradded. Provides stable regression targets during training.

**Predictor.** A 2-layer bidirectional transformer (4 heads, dim 256, MLP ratio 2) with learnable temporal position embeddings. Takes k encoded idea states and produces a prediction at position t+1. Approximately 500K-1M parameters.

**Loss.**
```
L = SmoothL1(z_pred, sg(z_target)) + lambda_reg * L_SigReg
```
SmoothL1 is preferred over L2 for robustness to outlier segments with large embedding shifts (following V-JEPA). SigReg (BCS loss) replaces VICReg and EMA-only approaches by directly constraining the embedding geometry toward an isotropic Gaussian.

## Benchmarks

| Dataset | Domain | Size | Task |
|---|---|---|---|
| PG-19 (Project Gutenberg) | Long-form narrative | 10K+ books, 3.5B words | Next-segment prediction, character arc dynamics |
| ROCStories | Short coherent narratives | 50K stories | Predict 5th sentence from first 4 |
| HellaSwag | Commonsense completion | 70K examples | Rank correct continuation above 3 distractors |
| ARC (AI2 Reasoning Challenge) | Science reasoning | 7.8K questions | Linear probe on learned representations |
| GSM8K | Grade school math | 8.5K step-by-step solutions | Linear probe for step-level reasoning dynamics |

## Evaluations

All comparisons use paired Wilcoxon signed-rank test. Bonferroni correction applied across the evaluation family. Results reported as mean +/- 95% CI across 5 random seeds.

| Eval | Metric | Baseline | Pass Criterion |
|---|---|---|---|
| Prediction accuracy | cos(z_pred, z_target) | copy-forward (z_t predicts z_{t+1}) | model > copy-forward, p < 0.05 |
| HellaSwag ranking | rank of correct continuation by distance | random (25%) | accuracy > copy-forward ranking |
| Long-horizon rollout | cos at k=1,2,4 steps ahead (autoregressive) | copy-forward at each horizon | stays above copy-forward at k >= 2 |
| Downstream reasoning | linear probe accuracy on ARC, GSM8K | frozen BGE-small encoder | beats frozen encoder on >= 2/4 probes, p < 0.05 |
| Representation quality | effective rank across corpus | n/a | rank > 0.3 * d |
| Calibration | ECE on confidence estimates | n/a | ECE < 0.1 |

Primary metric: prediction accuracy vs copy-forward baseline (Eval 1).

## Getting Started

```bash
conda create -n text-jepa python=3.10 pip
conda activate text-jepa
pip install -r requirements.txt
```

**Training (local):**
```bash
python src/main.py --fname configs/text_jepa_base.yaml --devices cuda:0
```

**Training (Modal, GPU):**
```bash
modal run experiments/train_text_jepa.py
```

**Evaluation:**
```bash
python experiments/eval_all.py --checkpoint results/checkpoints/best_model.pt
```

## Requirements

```
torch >= 2.0
transformers >= 4.40
sentence-transformers >= 2.2
datasets
numpy
scikit-learn
scipy
matplotlib
umap-learn
pytest
modal
```
