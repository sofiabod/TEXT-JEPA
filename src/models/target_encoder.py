import copy
import torch
import torch.nn as nn


class TargetEncoder(nn.Module):
    """ema copy of the context encoder. all outputs are stop-gradded via no_grad forward. update applies theta_target = m*theta_target + (1-m)*theta_online."""
    def __init__(self, online_encoder: nn.Module):
        super().__init__()
        self.encoder = copy.deepcopy(online_encoder)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder: nn.Module, m: float) -> None:
        for p_target, p_online in zip(
            self.encoder.parameters(), online_encoder.parameters()
        ):
            p_target.data.mul_(m).add_((1.0 - m) * p_online.detach().data)

    @torch.no_grad()
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(token_ids)
