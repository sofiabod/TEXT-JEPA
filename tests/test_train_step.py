import torch
import pytest


@pytest.fixture
def minimal_setup(tiny_encoder_cfg):
    from src.models.encoder import TextEncoder
    from src.models.target_encoder import TargetEncoder
    from src.models.predictor import TextJEPAPredictor
    from src.losses.prediction import TextJEPALoss
    online  = TextEncoder(**tiny_encoder_cfg)
    target  = TargetEncoder(online)
    pred    = TextJEPAPredictor(latent_dim=256, num_layers=2, num_heads=4, mlp_ratio=2, k=4,
                                temporal_stride=tiny_encoder_cfg["temporal_stride"])
    loss_fn = TextJEPALoss(lambda_reg=0.1)
    opt     = torch.optim.AdamW(
        list(online.parameters()) + list(pred.parameters()), lr=3e-4
    )
    return online, target, pred, loss_fn, opt


def test_online_encoder_params_change_after_step(minimal_setup, synthetic_token_batch):
    """after one training step, context encoder parameters must change."""
    online, target, pred, loss_fn, opt = minimal_setup
    params_before = {n: p.clone() for n, p in online.named_parameters()}

    tokens_ctx = synthetic_token_batch.unsqueeze(1).expand(-1, 4, -1)  # (B, k, L)
    k = 4
    z_ctx  = torch.stack([online(tokens_ctx[:, i]) for i in range(k)], dim=1)
    z_tgt  = target(synthetic_token_batch)
    z_pred = pred(z_ctx)
    loss   = loss_fn(z_pred, z_tgt)["total"]
    opt.zero_grad()
    loss.backward()
    opt.step()

    changed = any(
        not torch.allclose(p, params_before[n])
        for n, p in online.named_parameters()
    )
    assert changed, "online encoder params unchanged after training step"


def test_target_encoder_only_changes_via_ema(minimal_setup, synthetic_token_batch):
    """target encoder parameters must not change via backprop, only via ema update."""
    online, target, pred, loss_fn, opt = minimal_setup
    target_before = {n: p.clone() for n, p in target.named_parameters()}

    # training step without calling target.update
    tokens_ctx = synthetic_token_batch.unsqueeze(1).expand(-1, 4, -1)
    k = 4
    z_ctx  = torch.stack([online(tokens_ctx[:, i]) for i in range(k)], dim=1)
    z_tgt  = target(synthetic_token_batch)
    z_pred = pred(z_ctx)
    loss   = loss_fn(z_pred, z_tgt)["total"]
    opt.zero_grad()
    loss.backward()
    opt.step()

    for n, p in target.named_parameters():
        assert torch.allclose(p, target_before[n]), \
            f"target encoder param {n} changed via backprop; invariant violated"


def test_ema_update_changes_target_after_step(minimal_setup, synthetic_token_batch):
    """after calling target.update(), target params change toward online params."""
    online, target, pred, loss_fn, opt = minimal_setup
    target_before = {n: p.clone() for n, p in target.named_parameters()}

    # step and ema update
    tokens_ctx = synthetic_token_batch.unsqueeze(1).expand(-1, 4, -1)
    k = 4
    z_ctx  = torch.stack([online(tokens_ctx[:, i]) for i in range(k)], dim=1)
    z_tgt  = target(synthetic_token_batch)
    z_pred = pred(z_ctx)
    loss   = loss_fn(z_pred, z_tgt)["total"]
    opt.zero_grad()
    loss.backward()
    opt.step()
    target.update(online, m=0.99)

    changed = any(
        not torch.allclose(p, target_before[n])
        for n, p in target.named_parameters()
    )
    assert changed, "target encoder did not change after ema update"
