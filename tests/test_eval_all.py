import subprocess
import sys
import json
import pytest


def test_eval_all_script_is_valid_python():
    """experiments/eval_all.py must parse without syntax errors."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", "experiments/eval_all.py"],
        capture_output=True, text=True,
        cwd="/Users/sonia/TEXT-JEPA",
    )
    assert result.returncode == 0, f"syntax error:\n{result.stderr}"


def test_eval_all_script_has_checkpoint_arg():
    """eval_all.py must accept --checkpoint and --help must mention it."""
    result = subprocess.run(
        [sys.executable, "experiments/eval_all.py", "--help"],
        capture_output=True, text=True,
        cwd="/Users/sonia/TEXT-JEPA",
    )
    assert result.returncode == 0
    assert "--checkpoint" in result.stdout


def test_eval_all_runs_on_synthetic_checkpoint(tmp_path, tiny_encoder_cfg):
    """eval_all.py exits 0 and prints valid json results on a synthetic checkpoint."""
    import torch
    import yaml
    from src.models.encoder import TextEncoder
    from src.models.predictor import TextJEPAPredictor

    enc  = TextEncoder(**tiny_encoder_cfg)
    pred = TextJEPAPredictor(latent_dim=256, num_layers=2, num_heads=4,
                              mlp_ratio=2, k=4, temporal_stride=1)

    with open("experiments/configs/default.yaml") as f:
        config = yaml.safe_load(f)

    ckpt_path = tmp_path / "ck.pt"
    torch.save(
        {"encoder": enc.state_dict(), "predictor": pred.state_dict(),
         "seed": 0, "config": config},
        ckpt_path,
    )

    out_path = tmp_path / "results.json"
    result = subprocess.run(
        [sys.executable, "experiments/eval_all.py",
         "--checkpoint", str(ckpt_path),
         "--out", str(out_path),
         "--tiny"],
        capture_output=True, text=True,
        cwd="/Users/sonia/TEXT-JEPA",
    )
    assert result.returncode == 0, \
        f"eval_all failed:\n{result.stdout}\n{result.stderr}"

    data = json.loads(out_path.read_text())
    assert "eval1" in data
    assert "eval5" in data


def test_eval_all_prints_summary_table(tmp_path, tiny_encoder_cfg):
    """eval_all.py stdout contains a readable summary of eval results."""
    import torch
    import yaml
    from src.models.encoder import TextEncoder
    from src.models.predictor import TextJEPAPredictor

    enc  = TextEncoder(**tiny_encoder_cfg)
    pred = TextJEPAPredictor(latent_dim=256, num_layers=2, num_heads=4,
                              mlp_ratio=2, k=4, temporal_stride=1)

    with open("experiments/configs/default.yaml") as f:
        config = yaml.safe_load(f)

    ckpt_path = tmp_path / "ck.pt"
    torch.save(
        {"encoder": enc.state_dict(), "predictor": pred.state_dict(),
         "seed": 0, "config": config},
        ckpt_path,
    )

    result = subprocess.run(
        [sys.executable, "experiments/eval_all.py",
         "--checkpoint", str(ckpt_path),
         "--tiny"],
        capture_output=True, text=True,
        cwd="/Users/sonia/TEXT-JEPA",
    )
    assert result.returncode == 0
    assert "eval1" in result.stdout
    assert "eval5" in result.stdout
