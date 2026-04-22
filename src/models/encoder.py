import torch
import torch.nn as nn


class _TinyBackbone(nn.Module):
    """lightweight transformer for unit tests, runs on cpu, no hf download."""
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, max_seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos   = nn.Embedding(max_seq_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.hidden_dim = hidden_dim

    def forward(self, token_ids):
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.embed(token_ids) + self.pos(pos)
        x = self.transformer(x)
        return x.mean(dim=1)


class TextEncoder(nn.Module):
    """context encoder for text-jepa. backbone="tiny" uses cpu-only transformer; any hf model id loads from huggingface. all parameters train jointly. temporal_stride must come from config, never hardcoded."""
    def __init__(
        self,
        backbone: str = "tiny",
        latent_dim: int = 256,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        vocab_size: int = 128,
        max_seq_len: int = 512,
        temporal_stride: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.temporal_stride = temporal_stride
        if backbone == "tiny":
            self.backbone = _TinyBackbone(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
            )
            backbone_dim = hidden_dim
        else:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(backbone)
            backbone_dim = self.backbone.config.hidden_size
            for p in self.backbone.parameters():
                p.requires_grad_(True)

        self.proj = nn.Linear(backbone_dim, latent_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids is (B, L) long tensor; returns (B, latent_dim) float32 idea-state embedding."""
        if hasattr(self.backbone, "hidden_dim"):
            h = self.backbone(token_ids)
        else:
            out = self.backbone(input_ids=token_ids)
            h = out.last_hidden_state.mean(dim=1)
        return self.proj(h)
