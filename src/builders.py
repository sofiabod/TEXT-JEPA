from src.models.encoder import TextEncoder
from src.models.target_encoder import TargetEncoder
from src.models.predictor import TextJEPAPredictor
from src.losses.prediction import TextJEPALoss


def build_encoder(cfg: dict) -> TextEncoder:
    return TextEncoder(
        backbone=cfg.get("backbone") or cfg.get("encoder_backbone", "tiny"),
        latent_dim=cfg.get("latent_dim", 256),
        hidden_dim=cfg.get("hidden_dim", 64),
        num_layers=cfg.get("num_layers", 2),
        num_heads=cfg.get("num_heads", 2),
        vocab_size=cfg.get("vocab_size", 100),
        max_seq_len=cfg.get("max_seq_len", 512),
        temporal_stride=cfg["temporal_stride"],
    )


def build_target_encoder(online_encoder: TextEncoder) -> TargetEncoder:
    return TargetEncoder(online_encoder)


def build_predictor(cfg: dict) -> TextJEPAPredictor:
    return TextJEPAPredictor(
        latent_dim=cfg.get("latent_dim", 256),
        num_layers=cfg.get("predictor_layers", 2),
        num_heads=cfg.get("predictor_heads", 4),
        mlp_ratio=cfg.get("predictor_mlp_ratio", 2),
        k=cfg.get("context_window_k", 4),
        temporal_stride=cfg["temporal_stride"],
    )


def build_loss(cfg: dict) -> TextJEPALoss:
    return TextJEPALoss(
        lambda_reg=cfg.get("lambda_reg", 1.0),
        bcs_num_slices=cfg.get("bcs_num_slices", 1024),
        bcs_lmbd=cfg.get("bcs_lmbd", 0.1),
    )
