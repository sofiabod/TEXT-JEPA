import numpy as np
import torch
import pytest


# baselines

def test_copy_forward_returns_last_context():
    """copy-forward baseline returns the last context embedding."""
    from src.eval.baselines import copy_forward_predictions
    z_ctx = torch.randn(4, 5, 256)  # (B, k, d)
    out = copy_forward_predictions(z_ctx)
    assert out.shape == (4, 256)
    assert torch.allclose(out, z_ctx[:, -1, :])


def test_zero_baseline_returns_zeros():
    """zero baseline returns all-zero tensor of correct shape."""
    from src.eval.baselines import zero_baseline_predictions
    z_ctx = torch.randn(4, 5, 256)
    out = zero_baseline_predictions(z_ctx)
    assert out.shape == (4, 256)
    assert torch.all(out == 0)


# metrics

def test_cosine_similarity_batch_shape():
    """cosine_similarity_batch returns (B,) tensor."""
    from src.eval.metrics import cosine_similarity_batch
    a = torch.randn(8, 256)
    b = torch.randn(8, 256)
    sim = cosine_similarity_batch(a, b)
    assert sim.shape == (8,)


def test_cosine_similarity_identical_vectors_is_one():
    """cosine similarity of a vector with itself is 1.0."""
    from src.eval.metrics import cosine_similarity_batch
    a = torch.randn(4, 256)
    sim = cosine_similarity_batch(a, a)
    assert torch.allclose(sim, torch.ones(4), atol=1e-5)


def test_l2_on_sphere_identical_vectors_is_zero():
    """l2 distance between identical vectors on unit sphere is 0."""
    from src.eval.metrics import l2_on_sphere_batch
    z = torch.randn(4, 32)
    dists = l2_on_sphere_batch(z, z)
    assert torch.allclose(dists, torch.zeros(4), atol=1e-5)


def test_l2_on_sphere_orthogonal_vectors_is_sqrt2():
    """l2 distance between two orthogonal unit vectors is sqrt(2)."""
    from src.eval.metrics import l2_on_sphere_batch
    z1 = torch.zeros(1, 4); z1[0, 0] = 1.0
    z2 = torch.zeros(1, 4); z2[0, 1] = 1.0
    dist = l2_on_sphere_batch(z1, z2)
    assert abs(dist.item() - (2 ** 0.5)) < 1e-5


def test_effective_rank_diverse_embeddings_high():
    """random embeddings with n >> d have effective rank well above 0.3*d."""
    from src.eval.metrics import effective_rank
    z = torch.randn(512, 256)  # n >> d so rank is not bounded by n
    erank = effective_rank(z)
    assert erank > 0.3 * 256, f"expected erank > 76.8, got {erank:.2f}"


def test_effective_rank_collapsed_embeddings_low():
    """identical embeddings have effective rank near 1."""
    from src.eval.metrics import effective_rank
    z = torch.ones(64, 256)
    erank = effective_rank(z)
    assert erank < 5, f"expected erank near 1 for collapsed embeddings, got {erank:.2f}"


def test_wilcoxon_bonferroni_significant_difference():
    """wilcoxon with alternative='greater' (cosine similarity, higher is better) returns significant when model clearly beats baseline."""
    from src.eval.metrics import wilcoxon_bonferroni
    rng = np.random.default_rng(0)
    model    = rng.uniform(0.7, 0.9, 30)
    baseline = rng.uniform(0.1, 0.3, 30)
    result = wilcoxon_bonferroni(model, baseline, n_comparisons=6, alternative="greater")
    assert result["significant"], "expected significant result when model >> baseline"
    assert result["p_value_corrected"] < 0.05


def test_wilcoxon_bonferroni_no_difference():
    """wilcoxon with alternative='greater' (cosine similarity, higher is better) returns not significant when scores are nearly equal."""
    from src.eval.metrics import wilcoxon_bonferroni
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.5, 0.6, 30)
    result = wilcoxon_bonferroni(scores, scores + 0.001, n_comparisons=6, alternative="greater")
    assert not result["significant"]


def test_wilcoxon_bonferroni_less_significant():
    """wilcoxon with alternative='less' (l2 distance, lower is better) returns significant when model scores are clearly lower than baseline."""
    from src.eval.metrics import wilcoxon_bonferroni
    rng = np.random.default_rng(0)
    model_scores    = rng.uniform(0.1, 0.3, 50)   # clearly lower l2 distances
    baseline_scores = rng.uniform(0.7, 0.9, 50)   # clearly higher l2 distances
    result = wilcoxon_bonferroni(model_scores, baseline_scores, n_comparisons=1, alternative="less")
    assert result["significant"], "model with clearly lower l2 should be significant with alternative='less'"


def test_ci_95_returns_ordered_bounds():
    """95% ci lower bound < mean < upper bound."""
    from src.eval.metrics import ci_95
    rng = np.random.default_rng(0)
    values = rng.uniform(0.5, 0.8, 50)
    lo, hi = ci_95(values)
    assert lo < values.mean() < hi, \
        f"ci bounds wrong: lo={lo:.4f} mean={values.mean():.4f} hi={hi:.4f}"
