"""6 evals per agent-context.md. all run on held-out test set. call run_all_evals(encoder, predictor, test_dataset, device) to get results dict."""
import numpy as np
import torch

from src.eval.baselines import copy_forward_predictions
from src.eval.metrics import (
    cosine_similarity_batch, l2_on_sphere_batch, effective_rank, wilcoxon_bonferroni, ci_95
)


@torch.no_grad()
def eval1_prediction_accuracy(encoder, predictor, test_loader, device, k):
    """eval 1: does predictor beat copy-forward on l2 distance to target on unit sphere? lower is better."""
    model_dists, baseline_dists = [], []
    for batch in test_loader:
        tokens_ctx = batch["context_tokens"].to(device)
        tokens_tgt = batch["target_tokens"].to(device)

        z_ctx  = torch.stack([encoder(tokens_ctx[:, i]) for i in range(k)], dim=1)
        z_tgt  = encoder(tokens_tgt)
        z_pred = predictor(z_ctx)
        z_base = copy_forward_predictions(z_ctx)

        model_dists.extend(l2_on_sphere_batch(z_pred, z_tgt).cpu().numpy())
        baseline_dists.extend(l2_on_sphere_batch(z_base, z_tgt).cpu().numpy())

    model_dists    = np.array(model_dists)
    baseline_dists = np.array(baseline_dists)
    # 4 statistical tests: eval1, eval3 (all horizons as 1), eval5 erank threshold, eval6 ece threshold
    stats = wilcoxon_bonferroni(model_dists, baseline_dists, n_comparisons=4, alternative="less")
    lo, hi = ci_95(model_dists)
    return {
        "eval": 1,
        "model_mean_l2":    float(model_dists.mean()),
        "baseline_mean_l2": float(baseline_dists.mean()),
        "ci_95":            (lo, hi),
        **stats,
    }


@torch.no_grad()
def eval3_long_horizon_rollout(encoder, predictor, test_loader, device, k, max_steps=4):
    """eval 3: does prediction degrade gracefully at k=1,2,4 steps? autoregressive rollout feeds predicted embedding back as input."""
    results = {}
    for horizon in range(1, max_steps + 1):
        model_sims, baseline_sims = [], []
        for batch in test_loader:
            tokens_ctx     = batch["context_tokens"].to(device)
            tokens_tgt_seq = batch["future_tokens"].to(device)

            z_ctx = torch.stack([encoder(tokens_ctx[:, i]) for i in range(k)], dim=1)

            z_rolling = z_ctx.clone()
            for _ in range(horizon):
                z_step = predictor(z_rolling).unsqueeze(1)
                z_rolling = torch.cat([z_rolling[:, 1:], z_step], dim=1)

            z_tgt_h = encoder(tokens_tgt_seq[:, horizon - 1])
            z_base  = copy_forward_predictions(z_ctx)

            model_sims.extend(
                cosine_similarity_batch(z_rolling[:, -1], z_tgt_h).cpu().numpy()
            )
            baseline_sims.extend(
                cosine_similarity_batch(z_base, z_tgt_h).cpu().numpy()
            )

        model_sims    = np.array(model_sims)
        baseline_sims = np.array(baseline_sims)
        results[f"horizon_{horizon}"] = {
            "model_mean_cos":    float(model_sims.mean()),
            "baseline_mean_cos": float(baseline_sims.mean()),
            "model_beats_baseline": float(model_sims.mean()) > float(baseline_sims.mean()),
        }
    results["eval"] = 3
    return results


@torch.no_grad()
def eval5_representation_quality(encoder, test_loader, device):
    """eval 5: are representations non-collapsed? pass criterion is effective rank > 0.3 * d."""
    all_z = []
    for batch in test_loader:
        tokens = batch["target_tokens"].to(device)
        z = encoder(tokens)
        all_z.append(z.cpu())

    if not all_z:
        return {"eval": 5, "error": "no batches — loader empty"}
    all_z = torch.cat(all_z, dim=0)
    d     = all_z.shape[1]
    erank = effective_rank(all_z)
    return {
        "eval":           5,
        "effective_rank": erank,
        "latent_dim":     d,
        "threshold":      0.3 * d,
        "pass":           erank > 0.3 * d,
    }


@torch.no_grad()
def eval6_calibration(encoder, predictor, test_loader, device, k):
    """eval 6: ece (expected calibration error) < 0.1. treat cosine similarity as confidence, bucket by decile."""
    confidences, correct_flags = [], []
    for batch in test_loader:
        tokens_ctx = batch["context_tokens"].to(device)
        tokens_tgt = batch["target_tokens"].to(device)

        z_ctx  = torch.stack([encoder(tokens_ctx[:, i]) for i in range(k)], dim=1)
        z_tgt  = encoder(tokens_tgt)
        z_pred = predictor(z_ctx)
        z_base = copy_forward_predictions(z_ctx)

        conf     = cosine_similarity_batch(z_pred, z_tgt).cpu().numpy()
        base_cos = cosine_similarity_batch(z_base, z_tgt).cpu().numpy()
        flags    = (conf > base_cos).astype(float)
        confidences.extend(conf.tolist())
        correct_flags.extend(flags.tolist())

    confidences   = np.array(confidences)
    correct_flags = np.array(correct_flags)

    # ece: bin into 10 buckets
    bins = np.linspace(confidences.min(), confidences.max() + 1e-8, 11)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc  = correct_flags[mask].mean()
        ece += mask.mean() * abs(avg_conf - avg_acc)

    return {"eval": 6, "ece": float(ece), "pass": ece < 0.1}


@torch.no_grad()
def eval4_linear_probe(encoder, tokenizer, max_len: int, device: str) -> dict:
    """eval 4: linear probe accuracy on arc-easy, arc-challenge, gsm8k with frozen encoder."""
    from src.eval.linear_probe import run_eval4
    return run_eval4(encoder, tokenizer, max_len, device)


def run_all_evals(encoder, predictor, test_data, device, k,
                  tokenizer=None, max_len: int = 512):
    results     = {}
    test_loader = test_data.get("loader")
    results["eval1"] = eval1_prediction_accuracy(encoder, predictor, test_loader, device, k)
    if "future_loader" in test_data:
        results["eval3"] = eval3_long_horizon_rollout(
            encoder, predictor, test_data["future_loader"], device, k
        )
    results["eval5"] = eval5_representation_quality(encoder, test_loader, device)
    results["eval6"] = eval6_calibration(encoder, predictor, test_loader, device, k)
    if tokenizer is not None:
        results["eval4"] = eval4_linear_probe(encoder, tokenizer, max_len, device)
    return results
