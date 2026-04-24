"""run all evals on modal gpu: modal run experiments/modal_eval.py"""
import modal

VOLUME_NAME = "text-jepa-checkpoints"
IMAGE_NAME  = "text-jepa-eval"

text_jepa_image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.1",
        "transformers>=4.40",
        "datasets>=2.18",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
    )
    .add_local_python_source("src")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app    = modal.App(name=IMAGE_NAME, image=text_jepa_image)


@app.function(
    gpu="A10G",
    volumes={"/checkpoints": volume},
    timeout=3600,
)
def eval_seed(seed: int):
    import re
    import json
    import logging
    import torch
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset as hf_load

    logging.getLogger("transformers").setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings("ignore", message="enable_nested_tensor")

    from src.models.encoder import TextEncoder
    from src.models.predictor import TextJEPAPredictor
    from src.eval.evals import run_all_evals

    ck  = torch.load(f"/checkpoints/checkpoint_seed{seed}.pt", map_location="cpu")
    cfg = ck["config"]

    model_cfg = cfg["model"]
    enc = TextEncoder(
        backbone=model_cfg.get("encoder_backbone", "facebook/opt-125m"),
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
    enc.load_state_dict(ck["encoder"])
    pred.load_state_dict(ck["predictor"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc.to(device).eval()
    pred.to(device).eval()

    k       = cfg["data"].get("context_window_k", 4)
    max_len = cfg["model"].get("max_seq_len", 512)

    from transformers import AutoTokenizer
    backbone  = model_cfg.get("encoder_backbone", "facebook/opt-125m")
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _split(text):
        return re.split(r'(?<=[.!?])\s+', text.strip())

    def _tok(text):
        return tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

    class _DS(Dataset):
        def __init__(self):
            hf_ds = hf_load("mintujupally/ROCStories", split="test")
            self.windows = []
            for row in hf_ds:
                segs = _split(row["text"])
                for i in range(len(segs) - k):
                    self.windows.append({
                        "context": segs[i:i + k],
                        "target":  segs[i + k],
                        "future":  segs[i + 1:i + k + 1],
                    })

        def __len__(self): return len(self.windows)

        def __getitem__(self, idx):
            w = self.windows[idx]
            return {
                "context_tokens": torch.stack([_tok(s) for s in w["context"]]),
                "target_tokens":  _tok(w["target"]),
                "future_tokens":  torch.stack([_tok(s) for s in w["future"]]),
            }

    print(f"seed {seed}: loading test set...")
    ds     = _DS()
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    print(f"seed {seed}: {len(ds)} windows, running evals...")

    results = run_all_evals(enc, pred, {"loader": loader, "future_loader": loader},
                            device=device, k=k,
                            tokenizer=tokenizer, max_len=max_len)

    out_path = f"/checkpoints/results_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"seed {seed}: saved to {out_path}")

    # print summary
    if "eval1" in results:
        e = results["eval1"]
        print(f"  eval1  model={e['model_mean_l2']:.4f}  baseline={e['baseline_mean_l2']:.4f}  sig={e.get('significant')}")
    if "eval3" in results:
        e = results["eval3"]
        for horizon_key, v in e.items():
            if horizon_key == "eval":
                continue
            print(f"  eval3  {horizon_key}: model={v['model_mean_cos']:.4f}  beats_baseline={v.get('model_beats_baseline')}")
    if "eval4" in results:
        e = results["eval4"]
        for name in ("arc_easy", "arc_challenge", "gsm8k"):
            if name in e:
                v = e[name]
                print(f"  eval4  {name}: acc={v['accuracy']:.4f}  pass={v['pass']}")
    if "eval5" in results:
        e = results["eval5"]
        print(f"  eval5  erank={e.get('effective_rank', '?'):.1f}  pass={e.get('pass')}")
    if "eval6" in results:
        e = results["eval6"]
        print(f"  eval6  ece={e['ece']:.4f}  pass={e['pass']}")

    return results


@app.local_entrypoint()
def main():
    import yaml
    with open("experiments/configs/default.yaml") as f:
        config = yaml.safe_load(f)
    seeds = config["seeds"]
    for result in eval_seed.map(seeds):
        pass
