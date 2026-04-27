import modal

VOLUME_NAME = "text-jepa-checkpoints"
IMAGE_NAME  = "text-jepa-train"

text_jepa_image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.1",
        "transformers>=4.40",
        "datasets>=2.18",
        "numpy",
        "scipy",
        "pandas",
    )
    .add_local_python_source("src")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app    = modal.App(name=IMAGE_NAME, image=text_jepa_image)


@app.function(
    gpu="A10G",
    volumes={"/checkpoints": volume},
    timeout=3600 * 4,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_seed(seed: int, config: dict, dataset_name: str = "rocstories"):
    import re
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer
    from datasets import load_dataset as hf_load

    import warnings
    warnings.filterwarnings("ignore", message="enable_nested_tensor")
    from src.data.pg19 import PG19SegmentDataset
    from src.data.collator import ContextWindowCollator
    from src.train import train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    k       = config["data"]["context_window_k"]
    max_len = config["model"].get("max_seq_len", 128)
    backbone = config["model"].get("encoder_backbone", "facebook/opt-125m")

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tok(text):
        return tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

    def _split(text):
        return re.split(r'(?<=[.!?])\s+', text.strip())

    if dataset_name == "rocstories":
        hf_ds    = hf_load("mintujupally/ROCStories", split="train")
        collator = ContextWindowCollator(k=k)
        windows  = []
        for row in hf_ds:
            segs = _split(row["text"])
            windows.extend(collator(segs))
    else:
        raise NotImplementedError("pg19 tokenization not wired here yet")

    class _WindowDataset(Dataset):
        def __len__(self): return len(windows)
        def __getitem__(self, idx):
            w = windows[idx]
            return {
                "context_tokens": torch.stack([_tok(s) for s in w["context"]]),
                "target_tokens":  _tok(w["target"]),
            }

    full_ds   = _WindowDataset()
    n_total   = len(full_ds)
    n_train   = int(n_total * 0.70)
    n_val     = int(n_total * 0.10)

    train_ds = torch.utils.data.Subset(full_ds, range(0, n_train))
    val_ds   = torch.utils.data.Subset(full_ds, range(n_train, n_train + n_val))

    loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    enc, pred = train(config=config, loader=loader, device=device, seed=seed, val_loader=val_loader)

    torch.save(
        {
            "encoder":   enc.state_dict(),
            "predictor": pred.state_dict(),
            "seed":      seed,
            "config":    config,
        },
        f"/checkpoints/checkpoint_seed{seed}.pt",
    )
    print(f"seed {seed} complete, saved to /checkpoints/checkpoint_seed{seed}.pt")


@app.local_entrypoint()
def main():
    import yaml
    with open("experiments/configs/default.yaml") as f:
        config = yaml.safe_load(f)
    seeds = config["seeds"]
    for _ in train_seed.map(seeds, kwargs={"config": config}):
        pass
