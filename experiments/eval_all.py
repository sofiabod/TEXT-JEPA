"""run all evals on a checkpoint: python experiments/eval_all.py --checkpoint ck.pt"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from torch.utils.data import DataLoader, Dataset

from src.models.encoder import TextEncoder
from src.models.predictor import TextJEPAPredictor
from src.eval.evals import run_all_evals


def _split_sentences(text):
    import re
    return re.split(r'(?<=[.!?])\s+', text.strip())


class _ROCStoriesDataset(Dataset):
    """rocstories test split, tokenized with the same tokenizer used at training time."""
    def __init__(self, k, max_len, tok_fn):
        from datasets import load_dataset as hf_load
        hf_ds = hf_load("mintujupally/ROCStories", split="test")
        self.k       = k
        self.max_len = max_len
        self.tok_fn  = tok_fn
        self.windows = []
        for row in hf_ds:
            segs = _split_sentences(row["text"])
            for i in range(len(segs) - k):
                self.windows.append({
                    "context": segs[i:i + k],
                    "target":  segs[i + k],
                    "future":  segs[i + 1:i + k + 1],
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        ctx    = torch.stack([self.tok_fn(s) for s in w["context"]])
        tgt    = self.tok_fn(w["target"])
        future = torch.stack([self.tok_fn(s) for s in w["future"]])
        return {"context_tokens": ctx, "target_tokens": tgt, "future_tokens": future}


class _SyntheticDataset(Dataset):
    """tiny synthetic dataset for --tiny smoke tests only."""
    def __init__(self, vocab_size=100, seq_len=16, k=4, n=32):
        self.vocab = vocab_size
        self.seq   = seq_len
        self.k     = k
        self.n     = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "context_tokens": torch.randint(0, self.vocab, (self.k, self.seq)),
            "target_tokens":  torch.randint(0, self.vocab, (self.seq,)),
            "future_tokens":  torch.randint(0, self.vocab, (self.k, self.seq)),
        }


def _build_models(cfg, tiny=False):
    model_cfg = cfg["model"]
    if tiny:
        enc = TextEncoder(
            backbone="tiny", latent_dim=256, hidden_dim=64,
            num_layers=2, num_heads=2, vocab_size=100, max_seq_len=32,
            temporal_stride=model_cfg.get("temporal_stride", 1),
        )
    else:
        enc = TextEncoder(
            backbone=model_cfg.get("encoder_backbone", "tiny"),
            latent_dim=model_cfg["latent_dim"],
            temporal_stride=model_cfg.get("temporal_stride", 1),
        )
    pred = TextJEPAPredictor(
        latent_dim=model_cfg.get("latent_dim", 256),
        num_layers=model_cfg.get("predictor_layers", 2),
        num_heads=model_cfg.get("predictor_heads", 4),
        mlp_ratio=model_cfg.get("predictor_mlp_ratio", 2),
        k=cfg["data"].get("context_window_k", 4),
        temporal_stride=model_cfg.get("temporal_stride", 1),
    )
    return enc, pred


def _print_summary(results):
    print()
    print("eval results")
    print("-" * 48)
    if "eval1" in results:
        e = results["eval1"]
        print(f"eval1  prediction accuracy")
        print(f"       model l2:     {e['model_mean_l2']:.4f}")
        print(f"       baseline l2:  {e['baseline_mean_l2']:.4f}")
        print(f"       significant:  {e.get('significant', '?')}")
    if "eval3" in results:
        e = results["eval3"]
        print(f"eval3  long-horizon rollout")
        for k, v in e.items():
            if k == "eval":
                continue
            beats = v.get("model_beats_baseline", "?")
            print(f"       {k}: model={v['model_mean_cos']:.4f}  beats_baseline={beats}")
    if "eval4" in results:
        e = results["eval4"]
        print(f"eval4  linear probe")
        for name in ("arc_easy", "arc_challenge", "gsm8k"):
            if name in e:
                v = e[name]
                print(f"       {name}: acc={v['accuracy']:.4f}  pass={v['pass']}")
    if "eval5" in results:
        e = results["eval5"]
        print(f"eval5  representation quality")
        print(f"       effective_rank: {e['effective_rank']:.1f}  threshold: {e['threshold']:.1f}  pass: {e['pass']}")
    if "eval6" in results:
        e = results["eval6"]
        print(f"eval6  calibration  ece={e['ece']:.4f}  pass={e['pass']}")
    print("-" * 48)
    print()


def main():
    parser = argparse.ArgumentParser(description="run text-jepa evals on a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint .pt file")
    parser.add_argument("--out",        default=None,  help="optional path to save results as json")
    parser.add_argument("--device",     default="cpu", help="device (cpu or cuda)")
    parser.add_argument("--tiny",       action="store_true",
                        help="use tiny synthetic data (smoke test only)")
    args = parser.parse_args()

    ck  = torch.load(args.checkpoint, map_location=args.device)
    cfg = ck["config"]

    enc, pred = _build_models(cfg, tiny=args.tiny)
    enc.load_state_dict(ck["encoder"])
    pred.load_state_dict(ck["predictor"])
    enc.to(args.device).eval()
    pred.to(args.device).eval()

    k       = cfg["data"].get("context_window_k", 4)
    max_len = cfg["model"].get("max_seq_len", 512)

    if args.tiny:
        tokenizer = None
        ds     = _SyntheticDataset(vocab_size=100, seq_len=16, k=k, n=32)
        loader = DataLoader(ds, batch_size=8)
        test_data = {"loader": loader, "future_loader": loader}
    else:
        from transformers import AutoTokenizer
        backbone  = cfg["model"].get("encoder_backbone", "facebook/opt-125m")
        tokenizer = AutoTokenizer.from_pretrained(backbone)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def _tok_fn(text):
            return tokenizer(
                text,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)

        print("loading rocstories test split from huggingface...")
        ds     = _ROCStoriesDataset(k=k, max_len=max_len, tok_fn=_tok_fn)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        test_data = {"loader": loader, "future_loader": loader}

    results = run_all_evals(enc, pred, test_data, device=args.device, k=k,
                            tokenizer=tokenizer, max_len=max_len)
    _print_summary(results)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"results saved to {args.out}")

    sys.exit(0)


if __name__ == "__main__":
    main()
