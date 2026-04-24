import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon


def cosine_similarity_batch(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """per-sample cosine similarity between predictions and targets, returns (B,) tensor."""
    return F.cosine_similarity(z_pred, z_target, dim=-1)


def l2_on_sphere_batch(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """per-sample l2 distance after normalizing both vectors to unit hypersphere. returns (B,) tensor in [0, 2]."""
    z_pred_n   = F.normalize(z_pred,   dim=-1)
    z_target_n = F.normalize(z_target, dim=-1)
    return torch.norm(z_pred_n - z_target_n, dim=-1)


def effective_rank(embeddings: torch.Tensor) -> float:
    """effective rank = exp(entropy of normalized singular value spectrum). roy & vetterli 2007."""
    with torch.no_grad():
        s = torch.linalg.svdvals(embeddings.float())
        s = s / s.sum()
        s = s[s > 1e-10]
        entropy = -(s * torch.log(s)).sum()
        return float(torch.exp(entropy))


def wilcoxon_bonferroni(
    model_scores: np.ndarray,
    baseline_scores: np.ndarray,
    n_comparisons: int = 6,
    alternative: str = "less",
) -> dict:
    """paired wilcoxon signed-rank test with bonferroni correction. alternative='less' when lower model score = better (l2 distance)."""
    stat, p = wilcoxon(model_scores, baseline_scores, alternative=alternative)
    p_corrected = min(p * n_comparisons, 1.0)
    return {
        "statistic":         float(stat),
        "p_value_raw":       float(p),
        "p_value_corrected": float(p_corrected),
        "significant":       p_corrected < 0.05,
    }


def ci_95(values: np.ndarray) -> tuple[float, float]:
    """95% confidence interval via bootstrap, 1000 resamples."""
    rng  = np.random.default_rng(0)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(1000)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)
