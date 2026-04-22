import subprocess
import sys
import pytest


def test_redflag_script_is_valid_python():
    """experiments/redflag.py must parse without syntax errors."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", "experiments/redflag.py"],
        capture_output=True, text=True,
        cwd="/Users/sonia/TEXT-JEPA",
    )
    assert result.returncode == 0, f"syntax error:\n{result.stderr}"


def test_redflag_checks_all_8_invariants():
    """redflag.py source must contain all 8 invariant check labels."""
    with open("experiments/redflag.py") as f:
        source = f.read()
    labels = [
        "no token reconstruction",
        "encoder trained jointly",
        "target encoder ema only",
        "stop-gradient on target",
        "sigreg from eb_jepa",
        "predictor input is latent",
        "masking at predictor stage",
        "decoder decoupled",
    ]
    for label in labels:
        assert label in source, f"invariant label missing: {label}"


def test_redflag_runs_on_synthetic_checkpoint(tmp_path, tiny_encoder_cfg):
    """redflag.py must exit 0 when all invariants pass on a synthetic checkpoint."""
    import torch
    import yaml
    from src.models.encoder import TextEncoder
    from src.models.predictor import TextJEPAPredictor

    enc  = TextEncoder(**tiny_encoder_cfg)
    pred = TextJEPAPredictor(latent_dim=256, num_layers=2, num_heads=4,
                              mlp_ratio=2, k=4, temporal_stride=1)

    with open("experiments/configs/default.yaml") as f:
        config = yaml.safe_load(f)

    ckpt_path = tmp_path / "checkpoint_test.pt"
    torch.save(
        {"encoder": enc.state_dict(), "predictor": pred.state_dict(),
         "seed": 0, "config": config},
        ckpt_path,
    )

    result = subprocess.run(
        [sys.executable, "experiments/redflag.py", str(ckpt_path)],
        capture_output=True, text=True,
        cwd="/Users/sonia/TEXT-JEPA",
    )
    assert result.returncode == 0, \
        f"redflag failed on clean checkpoint:\n{result.stdout}\n{result.stderr}"
    assert "ALL PASS" in result.stdout
