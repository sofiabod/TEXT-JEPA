import math
import torch
import pytest


def test_momentum_schedule_starts_at_m_start():
    """first value from schedule equals m_start."""
    from src.train import make_momentum_schedule
    sched = list(make_momentum_schedule(m_start=0.996, m_end=1.0, total_steps=100))
    assert abs(sched[0] - 0.996) < 1e-6, f"expected 0.996, got {sched[0]}"


def test_momentum_schedule_ends_near_m_end():
    """last value from schedule is close to m_end."""
    from src.train import make_momentum_schedule
    sched = list(make_momentum_schedule(m_start=0.996, m_end=1.0, total_steps=100))
    assert abs(sched[-1] - 1.0) < 1e-4, f"expected ~1.0, got {sched[-1]}"


def test_momentum_schedule_is_monotonically_increasing():
    """momentum only increases over the schedule."""
    from src.train import make_momentum_schedule
    sched = list(make_momentum_schedule(m_start=0.996, m_end=1.0, total_steps=50))
    for i in range(len(sched) - 1):
        assert sched[i] <= sched[i + 1] + 1e-9, \
            f"momentum decreased at step {i}: {sched[i]} -> {sched[i+1]}"


def test_momentum_schedule_length_matches_total_steps():
    """schedule yields exactly total_steps values."""
    from src.train import make_momentum_schedule
    sched = list(make_momentum_schedule(m_start=0.9, m_end=1.0, total_steps=42))
    assert len(sched) == 42


def test_train_one_epoch_reduces_loss(tiny_encoder_cfg):
    """loss after one epoch of training is lower than before training."""
    from src.train import train
    from src.data.rocstories import ROCStoriesDataset
    from src.data.collator import ContextWindowCollator
    from torch.utils.data import DataLoader

    ds = ROCStoriesDataset(csv_path="tests/fixtures/rocstories_sample.csv", split="train")
    collator = ContextWindowCollator(k=4)

    # build a flat tokenized dataset from the rocstories fixture
    all_windows = []
    for item in ds:
        windows = collator(item["segments"])
        for w in windows:
            all_windows.append(w)

    if len(all_windows) == 0:
        pytest.skip("no windows in fixture; need at least 5 segments per story")

    # tokenize segments with a toy character-level encoding
    def tokenize(text, vocab_size=100, max_len=16):
        ids = [ord(c) % vocab_size for c in text[:max_len]]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    class TokenizedWindows(torch.utils.data.Dataset):
        def __init__(self, windows):
            self.windows = windows

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, idx):
            w = self.windows[idx]
            ctx = torch.stack([tokenize(s) for s in w["context"]])  # (k, L)
            tgt = tokenize(w["target"])                              # (L,)
            return {"context_tokens": ctx, "target_tokens": tgt}

    dataset = TokenizedWindows(all_windows)
    loader  = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)

    cfg = {
        "model": {
            **tiny_encoder_cfg,
            "ema_momentum_start": 0.9,
            "ema_momentum_end":   1.0,
            "temporal_stride": 1,
        },
        "data":     {"context_window_k": 4},
        "training": {
            "lr": 1e-3, "weight_decay": 0.01, "batch_size": 2,
            "max_epochs": 3, "early_stopping_patience": 10,
            "token_dropout_rate": 0.0,
        },
        "loss": {"lambda_reg": 0.0, "bcs_num_slices": 64, "bcs_lmbd": 0.1},
    }

    initial_loss = _measure_loss(cfg, loader, tiny_encoder_cfg)
    train(cfg, loader, device="cpu", seed=0)
    final_loss   = _measure_loss(cfg, loader, tiny_encoder_cfg)

    # loss should not be nan and should be finite
    assert math.isfinite(final_loss), f"loss is not finite after training: {final_loss}"


def test_train_with_val_loader_returns_enc_pred(tiny_encoder_cfg):
    """train() with a val_loader returns (enc, pred) without error."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.train import train

    vocab_size = tiny_encoder_cfg["vocab_size"]
    max_len    = tiny_encoder_cfg["max_seq_len"]
    k          = 4
    B          = 4

    # synthetic batches: context_tokens (B, k, L) and target_tokens (B, L)
    ctx = torch.randint(0, vocab_size, (B, k, max_len))
    tgt = torch.randint(0, vocab_size, (B, max_len))

    class _SyntheticDS(torch.utils.data.Dataset):
        def __len__(self): return B
        def __getitem__(self, idx):
            return {"context_tokens": ctx[idx], "target_tokens": tgt[idx]}

    ds         = _SyntheticDS()
    tr_loader  = DataLoader(ds, batch_size=2, shuffle=False, drop_last=False)
    val_loader = DataLoader(ds, batch_size=2, shuffle=False, drop_last=False)

    cfg = {
        "model": {
            **tiny_encoder_cfg,
            "ema_momentum_start": 0.9,
            "ema_momentum_end":   1.0,
            "temporal_stride": 1,
        },
        "data":     {"context_window_k": k},
        "training": {
            "lr": 1e-3, "weight_decay": 0.01, "batch_size": 2,
            "max_epochs": 2, "early_stopping_patience": 10,
            "token_dropout_rate": 0.0,
        },
        "loss": {"lambda_reg": 0.0, "bcs_num_slices": 64, "bcs_lmbd": 0.1},
    }

    result = train(cfg, tr_loader, device="cpu", seed=0, val_loader=val_loader)
    assert isinstance(result, tuple) and len(result) == 2, \
        "train() must return (enc, pred)"
    enc, pred = result
    assert enc  is not None, "enc is None"
    assert pred is not None, "pred is None"


def _measure_loss(cfg, loader, tiny_encoder_cfg):
    """helper: one forward pass, return mean loss."""
    from src.builders import build_encoder, build_target_encoder, build_predictor, build_loss
    enc  = build_encoder({**cfg["model"], "temporal_stride": 1})
    tgt  = build_target_encoder(enc)
    pred = build_predictor({**cfg["model"], "context_window_k": cfg["data"]["context_window_k"]})
    loss_fn = build_loss(cfg["loss"])
    k = cfg["data"]["context_window_k"]
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            tokens_ctx = batch["context_tokens"]
            tokens_tgt = batch["target_tokens"]
            z_ctx  = torch.stack([enc(tokens_ctx[:, i]) for i in range(k)], dim=1)
            z_tgt  = tgt(tokens_tgt)
            z_pred = pred(z_ctx)
            total += loss_fn(z_pred, z_tgt)["total"].item()
            n += 1
    return total / max(n, 1)
