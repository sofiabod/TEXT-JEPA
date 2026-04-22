import torch
import pytest


def test_predictor_output_shape():
    """predictor maps (batch, k, d) context and mask to (batch, d) prediction."""
    from src.models.predictor import TextJEPAPredictor
    pred = TextJEPAPredictor(latent_dim=256, num_layers=2, num_heads=4, mlp_ratio=2, k=4,
                              temporal_stride=1)
    z_context = torch.randn(3, 4, 256)
    z_pred = pred(z_context)
    assert z_pred.shape == (3, 256), f"expected (3, 256), got {z_pred.shape}"


def test_predictor_mask_token_is_learnable():
    """predictor has a learnable mask token parameter."""
    from src.models.predictor import TextJEPAPredictor
    pred = TextJEPAPredictor(latent_dim=256, num_layers=2, num_heads=4, mlp_ratio=2, k=4,
                              temporal_stride=1)
    mask_params = [n for n, p in pred.named_parameters() if "mask_token" in n]
    assert len(mask_params) == 1, f"expected 1 mask_token param, got {mask_params}"
    mask_p = dict(pred.named_parameters())[mask_params[0]]
    assert mask_p.requires_grad


def test_predictor_bidirectional_attention():
    """changing a context embedding at any position changes the prediction, verifying bidirectional (not causal) attention."""
    from src.models.predictor import TextJEPAPredictor
    pred = TextJEPAPredictor(latent_dim=256, num_layers=2, num_heads=4, mlp_ratio=2, k=4,
                              temporal_stride=1)
    pred.eval()
    z_base = torch.randn(1, 4, 256)
    with torch.no_grad():
        out_base = pred(z_base)
    z_perturbed = z_base.clone()
    z_perturbed[0, 0] += 1.0
    with torch.no_grad():
        out_perturbed = pred(z_perturbed)
    assert not torch.allclose(out_base, out_perturbed), \
        "perturbing context[0] should change output; check for causal masking bug"


def test_predictor_mask_token_in_input():
    """verifies the mask token is injected: full sequence length inside transformer should be k+1."""
    from src.models.predictor import TextJEPAPredictor
    pred = TextJEPAPredictor(latent_dim=256, num_layers=2, num_heads=4, mlp_ratio=2, k=4,
                              temporal_stride=1)
    original_forward = pred.transformer.forward
    captured = {}

    def patched_forward(x, *args, **kwargs):
        captured["seq_len"] = x.shape[1]
        return original_forward(x, *args, **kwargs)

    pred.transformer.forward = patched_forward
    z_context = torch.randn(2, 4, 256)
    pred(z_context)
    assert captured["seq_len"] == 5, \
        f"expected k+1=5 tokens in transformer, got {captured['seq_len']}"


def test_predictor_has_temporal_stride_param():
    """predictor exposes temporal_stride as a named constructor parameter."""
    from src.models.predictor import TextJEPAPredictor
    import inspect
    sig = inspect.signature(TextJEPAPredictor.__init__)
    assert "temporal_stride" in sig.parameters, \
        "TextJEPAPredictor must have explicit temporal_stride parameter"
    pred = TextJEPAPredictor(latent_dim=256, num_layers=2, num_heads=4, mlp_ratio=2, k=4,
                              temporal_stride=1)
    assert pred.temporal_stride == 1
