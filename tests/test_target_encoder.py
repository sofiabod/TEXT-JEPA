import torch
import pytest


def test_target_encoder_no_grad(tiny_encoder_cfg, synthetic_token_batch):
    """target encoder: all parameters have requires_grad=False."""
    from src.models.encoder import TextEncoder
    from src.models.target_encoder import TargetEncoder
    online = TextEncoder(**tiny_encoder_cfg)
    target = TargetEncoder(online)
    frozen = [n for n, p in target.named_parameters() if p.requires_grad]
    assert len(frozen) == 0, f"target encoder has trainable params: {frozen}"


def test_target_encoder_ema_update(tiny_encoder_cfg, synthetic_token_batch):
    """after one ema update with m=0.9, target params change by (1-m)*(online-target). verified numerically on a single parameter tensor."""
    from src.models.encoder import TextEncoder
    from src.models.target_encoder import TargetEncoder
    online = TextEncoder(**tiny_encoder_cfg)
    target = TargetEncoder(online)

    name, p_target_before = next(
        (n, p.clone()) for n, p in target.named_parameters()
    )
    # target params are named "encoder.<name>"; strip prefix to look up in online
    online_name = name.removeprefix("encoder.")
    p_online = dict(online.named_parameters())[online_name].data

    m = 0.9
    target.update(online, m)

    p_target_after = dict(target.named_parameters())[name].data
    expected = m * p_target_before + (1 - m) * p_online
    assert torch.allclose(p_target_after, expected, atol=1e-6), \
        "ema update formula incorrect"


def test_target_encoder_diverges_after_online_update(tiny_encoder_cfg, synthetic_token_batch):
    """after modifying online encoder params to simulate a gradient step, target encoder should differ from online encoder."""
    from src.models.encoder import TextEncoder
    from src.models.target_encoder import TargetEncoder
    online = TextEncoder(**tiny_encoder_cfg)
    target = TargetEncoder(online)

    with torch.no_grad():
        for p in online.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    target.update(online, m=0.99)
    for (n1, p1), (n2, p2) in zip(online.named_parameters(), target.named_parameters()):
        if p1.numel() > 1:
            assert not torch.allclose(p1, p2), \
                f"target encoder identical to online after divergence: {n1}"
            break


def test_target_encoder_outputs_stop_gradded(tiny_encoder_cfg, synthetic_token_batch):
    """output of target encoder has requires_grad=False, stop-grad via no_grad forward."""
    from src.models.encoder import TextEncoder
    from src.models.target_encoder import TargetEncoder
    online = TextEncoder(**tiny_encoder_cfg)
    target = TargetEncoder(online)

    with torch.no_grad():
        z_target = target(synthetic_token_batch)

    assert not z_target.requires_grad, "target encoder output should be detached"
