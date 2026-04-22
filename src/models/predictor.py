import torch
import torch.nn as nn


class TextJEPAPredictor(nn.Module):
    """2-layer bidirectional transformer predictor. takes (B, k, d) context embeddings, appends mask token, runs full attention, returns (B, d) prediction at mask position. temporal_stride must come from config."""
    def __init__(
        self,
        latent_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        k: int = 4,
        temporal_stride: int = 1,
    ):
        super().__init__()
        self.temporal_stride = temporal_stride
        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pos_emb = nn.Embedding(k + 1, latent_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * mlp_ratio,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(latent_dim)
        self.k = k

    def forward(self, z_context: torch.Tensor) -> torch.Tensor:
        """z_context is (B, k, d); returns z_pred (B, d) at mask position."""
        B = z_context.shape[0]
        mask = self.mask_token.expand(B, 1, -1)
        seq = torch.cat([z_context, mask], dim=1)  # (B, k+1, d)

        pos_ids = torch.arange(self.k + 1, device=seq.device)
        seq = seq + self.pos_emb(pos_ids)

        out = self.transformer(seq)
        out = self.norm(out)
        return out[:, -1, :]  # (B, d) mask position
