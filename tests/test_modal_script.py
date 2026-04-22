import subprocess
import sys
import pytest


def test_modal_train_script_is_valid_python():
    """experiments/modal_train.py must parse without syntax errors."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", "experiments/modal_train.py"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, \
        f"syntax error in modal_train.py:\n{result.stderr}"


def test_modal_train_script_imports_src_modules():
    """modal_train.py must reference the correct src module paths."""
    with open("experiments/modal_train.py") as f:
        source = f.read()
    assert "from src.train import train" in source
    assert "from src.data.rocstories import ROCStoriesDataset" in source
    assert "from src.data.pg19 import PG19SegmentDataset" in source


def test_modal_train_script_saves_correct_checkpoint_keys():
    """checkpoint dict saved by modal_train must include encoder, predictor, seed, config."""
    with open("experiments/modal_train.py") as f:
        source = f.read()
    for key in ('"encoder"', '"predictor"', '"seed"', '"config"'):
        assert key in source, f"checkpoint key {key} missing from modal_train.py"
