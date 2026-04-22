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
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app    = modal.App(name=IMAGE_NAME, image=text_jepa_image)


@app.function(
    gpu="A10G",
    volumes={"/checkpoints": volume},
    timeout=3600 * 4,
)
def train_seed(seed: int, config: dict, dataset_name: str = "rocstories"):
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from src.data.rocstories import ROCStoriesDataset
    from src.data.pg19 import PG19SegmentDataset
    from src.data.collator import ContextWindowCollator
    from src.train import train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    k = config["data"]["context_window_k"]

    if dataset_name == "rocstories":
        raw_ds = ROCStoriesDataset(
            csv_path="/checkpoints/rocstories.csv", split="train"
        )
        collator = ContextWindowCollator(k=k)
        windows = []
        for item in raw_ds:
            windows.extend(collator(item["segments"]))
        dataset = _WindowDataset(windows, config)
    else:
        dataset = PG19SegmentDataset(split="train")

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        drop_last=True,
    )

    enc, pred = train(config=config, loader=loader, device=device, seed=seed)

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


class _WindowDataset:
    """tokenized rocstories windows for modal training."""
    def __init__(self, windows, config):
        import torch
        self._windows = windows
        self._max_len = config["model"].get("max_seq_len", 512)

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        import torch
        w = self._windows[idx]

        def tok(text):
            ids = [ord(c) % 30000 for c in text[: self._max_len]]
            ids += [0] * (self._max_len - len(ids))
            return torch.tensor(ids, dtype=torch.long)

        ctx = torch.stack([tok(s) for s in w["context"]])
        tgt = tok(w["target"])
        return {"context_tokens": ctx, "target_tokens": tgt}


@app.local_entrypoint()
def main():
    import yaml
    with open("experiments/configs/default.yaml") as f:
        config = yaml.safe_load(f)
    seeds = config["seeds"]
    for _ in train_seed.map(seeds, kwargs={"config": config}):
        pass
