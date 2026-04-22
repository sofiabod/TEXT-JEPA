import torch
import pytest


def test_encoder_output_shape(tiny_encoder_cfg, synthetic_token_batch):
    """encoder maps token sequences to (batch, latent_dim=256) vectors."""
    from src.models.encoder import TextEncoder
    enc = TextEncoder(**tiny_encoder_cfg)
    z = enc(synthetic_token_batch)
    assert z.shape == (4, 256), f"expected (4, 256), got {z.shape}"


def test_encoder_all_params_trainable(tiny_encoder_cfg, synthetic_token_batch):
    """all encoder parameters must require grad, no frozen backbone."""
    from src.models.encoder import TextEncoder
    enc = TextEncoder(**tiny_encoder_cfg)
    frozen = [n for n, p in enc.named_parameters() if not p.requires_grad]
    assert len(frozen) == 0, f"frozen params found: {frozen}"


def test_encoder_no_token_reconstruction_loss(tiny_encoder_cfg, synthetic_token_batch):
    """encoder has no cross-entropy head, it only outputs latent vectors."""
    from src.models.encoder import TextEncoder
    enc = TextEncoder(**tiny_encoder_cfg)
    vocab_modules = [
        n for n, m in enc.named_modules()
        if hasattr(m, "out_features") and getattr(m, "out_features", 0) == tiny_encoder_cfg["vocab_size"]
    ]
    assert len(vocab_modules) == 0, f"vocab projection found: {vocab_modules}"


def test_encoder_output_dtype(tiny_encoder_cfg, synthetic_token_batch):
    """encoder output is float32 on cpu."""
    from src.models.encoder import TextEncoder
    enc = TextEncoder(**tiny_encoder_cfg)
    z = enc(synthetic_token_batch)
    assert z.dtype == torch.float32
    assert z.device.type == "cpu"


def test_encoder_has_temporal_stride_param(tiny_encoder_cfg):
    """encoder exposes temporal_stride as a named constructor parameter, not hardcoded."""
    from src.models.encoder import TextEncoder
    import inspect
    sig = inspect.signature(TextEncoder.__init__)
    assert "temporal_stride" in sig.parameters, \
        "TextEncoder must have explicit temporal_stride parameter"
    enc = TextEncoder(**tiny_encoder_cfg)
    assert enc.temporal_stride == 1
