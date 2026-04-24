import torch
import numpy as np
import pytest


@pytest.fixture
def tiny_enc(tiny_encoder_cfg):
    from src.models.encoder import TextEncoder
    return TextEncoder(**tiny_encoder_cfg)


def test_extract_embeddings_shape(tiny_enc):
    """extract_embeddings returns (N, latent_dim) numpy array for a list of (N, L) token tensors."""
    from src.eval.linear_probe import extract_embeddings
    tokens = [torch.randint(0, 100, (8, 16)) for _ in range(3)]  # 3 batches of 8
    embs = extract_embeddings(tiny_enc, tokens, device="cpu")
    assert embs.shape == (24, 256), f"expected (24, 256), got {embs.shape}"


def test_train_and_eval_probe_returns_accuracy():
    """train_probe + eval_probe return a float accuracy in [0, 1]."""
    from src.eval.linear_probe import train_probe, eval_probe
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((40, 256))
    y_train = np.arange(40) % 4
    X_test  = rng.standard_normal((20, 256))
    y_test  = np.arange(20) % 4
    probe   = train_probe(X_train, y_train)
    acc     = eval_probe(probe, X_test, y_test)
    assert 0.0 <= acc <= 1.0


def test_eval4_synthetic_returns_required_keys(tiny_enc):
    """eval4_linear_probe_synthetic returns arc_easy, arc_challenge, gsm8k each with accuracy and pass."""
    from src.eval.linear_probe import eval4_linear_probe_synthetic
    result = eval4_linear_probe_synthetic(tiny_enc, n_train=40, n_test=20, device="cpu")
    for key in ("arc_easy", "arc_challenge", "gsm8k"):
        assert key in result, f"missing key: {key}"
        assert "accuracy" in result[key]
        assert "pass" in result[key]
