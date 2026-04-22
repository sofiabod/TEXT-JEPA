import yaml
import pytest


@pytest.fixture
def default_cfg():
    with open("experiments/configs/default.yaml") as f:
        return yaml.safe_load(f)


def test_build_encoder_returns_textencoder(default_cfg, tiny_encoder_cfg):
    """build_encoder returns a TextEncoder with temporal_stride from config."""
    from src.builders import build_encoder
    from src.models.encoder import TextEncoder
    enc = build_encoder(tiny_encoder_cfg)
    assert isinstance(enc, TextEncoder)
    assert enc.temporal_stride == tiny_encoder_cfg["temporal_stride"]


def test_build_target_encoder_returns_targetencoder(tiny_encoder_cfg):
    """build_target_encoder returns a TargetEncoder wrapping the online encoder."""
    from src.builders import build_encoder, build_target_encoder
    from src.models.target_encoder import TargetEncoder
    online = build_encoder(tiny_encoder_cfg)
    target = build_target_encoder(online)
    assert isinstance(target, TargetEncoder)
    frozen = [n for n, p in target.named_parameters() if p.requires_grad]
    assert len(frozen) == 0, "TargetEncoder must have no trainable params"


def test_build_predictor_returns_predictor(default_cfg, tiny_encoder_cfg):
    """build_predictor returns a TextJEPAPredictor with temporal_stride from config."""
    from src.builders import build_predictor
    from src.models.predictor import TextJEPAPredictor
    cfg = {**default_cfg["model"], **{"temporal_stride": tiny_encoder_cfg["temporal_stride"]}}
    pred = build_predictor(cfg)
    assert isinstance(pred, TextJEPAPredictor)
    assert pred.temporal_stride == cfg["temporal_stride"]


def test_build_loss_returns_loss(default_cfg):
    """build_loss returns a TextJEPALoss with BCS using config defaults."""
    from src.builders import build_loss
    from src.losses.prediction import TextJEPALoss
    from src.losses.anticollapse import BCS
    loss_fn = build_loss(default_cfg["loss"])
    assert isinstance(loss_fn, TextJEPALoss)
    bcs = next(m for m in loss_fn.modules() if isinstance(m, BCS))
    assert bcs.num_slices == default_cfg["loss"]["bcs_num_slices"]
    assert bcs.lmbd == default_cfg["loss"]["bcs_lmbd"]


def test_builders_wire_temporal_stride_from_config(default_cfg, tiny_encoder_cfg):
    """temporal_stride flows from config into both encoder and predictor."""
    from src.builders import build_encoder, build_predictor
    cfg = {**default_cfg["model"], **tiny_encoder_cfg, "temporal_stride": 1}
    enc  = build_encoder(cfg)
    pred = build_predictor(cfg)
    assert enc.temporal_stride  == 1
    assert pred.temporal_stride == 1
