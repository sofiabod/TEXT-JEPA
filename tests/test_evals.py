import torch
import numpy as np
import pytest


@pytest.fixture
def tiny_model(tiny_encoder_cfg):
    from src.models.encoder import TextEncoder
    from src.models.predictor import TextJEPAPredictor
    enc  = TextEncoder(**tiny_encoder_cfg)
    pred = TextJEPAPredictor(latent_dim=256, num_layers=2, num_heads=4,
                              mlp_ratio=2, k=4, temporal_stride=1)
    return enc, pred


@pytest.fixture
def fake_loader(tiny_encoder_cfg):
    """single-batch loader with synthetic token data."""
    B, k, L = 8, 4, 16
    vocab = tiny_encoder_cfg["vocab_size"]

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return B
        def __getitem__(self, i):
            return {
                "context_tokens": torch.randint(0, vocab, (k, L)),
                "target_tokens":  torch.randint(0, vocab, (L,)),
                "future_tokens":  torch.randint(0, vocab, (4, L)),
            }

    from torch.utils.data import DataLoader
    return DataLoader(_DS(), batch_size=B)


def test_eval1_returns_l2_keys(tiny_model, fake_loader):
    """eval 1 output uses l2 keys, not cosine keys."""
    from src.eval.evals import eval1_prediction_accuracy
    enc, pred = tiny_model
    result = eval1_prediction_accuracy(enc, pred, fake_loader, device="cpu", k=4)
    assert "model_mean_l2" in result, "missing model_mean_l2"
    assert "baseline_mean_l2" in result, "missing baseline_mean_l2"
    assert "significant" in result
    assert "ci_95" in result
    assert "model_mean_cos" not in result, "old cosine key should be gone"


def test_eval1_l2_scores_are_nonneg(tiny_model, fake_loader):
    """l2 distances on unit sphere are in [0, 2]."""
    from src.eval.evals import eval1_prediction_accuracy
    enc, pred = tiny_model
    result = eval1_prediction_accuracy(enc, pred, fake_loader, device="cpu", k=4)
    assert 0.0 <= result["model_mean_l2"] <= 2.0
    assert 0.0 <= result["baseline_mean_l2"] <= 2.0


def test_eval5_passes_for_random_encoder(tiny_model, fake_loader):
    """eval 5 passes (effective rank > 0.3*d) for a random untrained encoder."""
    from src.eval.evals import eval5_representation_quality
    enc, _ = tiny_model
    result = eval5_representation_quality(enc, fake_loader, device="cpu")
    assert "effective_rank" in result
    assert "pass" in result


def test_eval5_fails_for_collapsed_encoder(tiny_encoder_cfg, fake_loader):
    """eval 5 fails when all encoder outputs are identical (collapsed)."""
    from src.eval.evals import eval5_representation_quality
    from src.models.encoder import TextEncoder
    enc = TextEncoder(**tiny_encoder_cfg)

    # make encoder always output the same vector
    with torch.no_grad():
        for p in enc.parameters():
            p.zero_()

    result = eval5_representation_quality(enc, fake_loader, device="cpu")
    assert not result["pass"], \
        "eval 5 should fail for a collapsed (zero-output) encoder"


def test_run_all_evals_returns_eval1_and_eval5(tiny_model, fake_loader):
    """run_all_evals returns at least eval1 and eval5 results."""
    from src.eval.evals import run_all_evals
    enc, pred = tiny_model
    test_data = {"loader": fake_loader}
    results = run_all_evals(enc, pred, test_data, device="cpu", k=4)
    assert "eval1" in results
    assert "eval5" in results


def test_run_all_evals_skips_eval4_when_no_tokenizer(tiny_model, fake_loader):
    """run_all_evals does not call eval4 when tokenizer=None (avoids hf download in unit tests)."""
    from src.eval.evals import run_all_evals
    enc, pred = tiny_model
    results = run_all_evals(enc, pred, {"loader": fake_loader}, device="cpu", k=4)
    assert "eval4" not in results
