import math
import torch
from torch.utils.data import DataLoader

from src.builders import build_encoder, build_target_encoder, build_predictor, build_loss


def make_momentum_schedule(m_start: float, m_end: float, total_steps: int):
    """cosine schedule for ema momentum from m_start to m_end over total_steps."""
    for i in range(total_steps):
        yield m_end - (m_end - m_start) * (math.cos(math.pi * i / total_steps) + 1) / 2


def train(config: dict, loader: DataLoader, device: str = "cuda", seed: int = 0, val_loader: DataLoader | None = None):
    torch.manual_seed(seed)

    model_cfg = {**config["model"], "temporal_stride": config["model"]["temporal_stride"]}
    k         = config["data"]["context_window_k"]

    enc     = build_encoder(model_cfg).to(device)
    tgt     = build_target_encoder(enc)
    pred    = build_predictor({**model_cfg, "context_window_k": k}).to(device)
    loss_fn = build_loss(config["loss"]).to(device)

    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(pred.parameters()),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    epochs   = config["training"]["max_epochs"]
    patience = config["training"]["early_stopping_patience"]

    total_steps = epochs * max(len(loader), 1)
    momentum_schedule = make_momentum_schedule(
        config["model"]["ema_momentum_start"],
        config["model"]["ema_momentum_end"],
        total_steps,
    )

    best_loss       = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        enc.train()
        pred.train()
        epoch_loss = 0.0

        for batch in loader:
            tokens_ctx = batch["context_tokens"].to(device)  # (B, k, L)
            tokens_tgt = batch["target_tokens"].to(device)   # (B, L)

            # token dropout on context only, training only; targets stay clean
            dropout_rate = config["training"]["token_dropout_rate"]
            mask = torch.bernoulli(
                torch.full(tokens_ctx.shape, dropout_rate, device=device)
            ).bool()
            tokens_ctx = tokens_ctx.masked_fill(mask, 0)

            # context encoder (gradients flow)
            z_ctx = torch.stack(
                [enc(tokens_ctx[:, i]) for i in range(k)], dim=1
            )

            # target encoder, no grad
            with torch.no_grad():
                z_tgt = tgt(tokens_tgt)

            z_pred = pred(z_ctx)
            out    = loss_fn(z_pred, z_tgt)
            loss   = out["total"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(pred.parameters()), 1.0
            )
            opt.step()

            m = next(momentum_schedule)
            tgt.update(enc, m)
            epoch_loss += loss.item()

        avg = epoch_loss / max(len(loader), 1)

        if val_loader is not None:
            enc.eval()
            pred.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    tokens_ctx = val_batch["context_tokens"].to(device)
                    tokens_tgt = val_batch["target_tokens"].to(device)
                    z_ctx = torch.stack(
                        [enc(tokens_ctx[:, i]) for i in range(k)], dim=1
                    )
                    z_tgt = tgt(tokens_tgt)
                    z_pred = pred(z_ctx)
                    val_loss += loss_fn(z_pred, z_tgt)["total"].item()
            monitor = val_loss / max(len(val_loader), 1)
        else:
            monitor = avg

        if monitor < best_loss:
            best_loss        = monitor
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return enc, pred
