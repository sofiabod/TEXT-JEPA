import pytest
import torch


@pytest.fixture
def tiny_encoder_cfg():
    """tiny encoder config that runs on cpu in under 2s, 2-layer transformer d=64 to 256."""
    return {
        "backbone": "tiny",
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "latent_dim": 256,
        "vocab_size": 100,
        "max_seq_len": 32,
        "temporal_stride": 1,
    }


@pytest.fixture
def synthetic_token_batch(tiny_encoder_cfg):
    """batch of 4 token sequences of length 16."""
    return torch.randint(0, tiny_encoder_cfg["vocab_size"], (4, 16))
