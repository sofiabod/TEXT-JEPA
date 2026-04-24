import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.anticollapse import BCS


class TextJEPALoss(nn.Module):
    """l = l2(normalize(z_pred), normalize(sg(z_target))) + lambda_reg * l_bcs. both embeddings projected to unit hypersphere before loss. bcs imported from src.losses.anticollapse, copied verbatim from eb_jepa hier_cost_exp."""
    def __init__(
        self,
        lambda_reg: float = 1.0,
        bcs_num_slices: int = 1024,
        bcs_lmbd: float = 0.1,
    ):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.bcs = BCS(num_slices=bcs_num_slices, lmbd=bcs_lmbd)

    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> dict:
        """z_pred and z_target are (B, d); z_target is detached. both normalized to unit hypersphere before loss. returns dict with total, smooth_l1, sigreg."""
        z_target_sg = z_target.detach()
        z_pred_n    = F.normalize(z_pred,      dim=-1)
        z_target_n  = F.normalize(z_target_sg, dim=-1)
        l2        = F.mse_loss(z_pred_n, z_target_n)
        bcs_out   = self.bcs(z_pred_n, z_target_n)
        sigreg    = bcs_out["bcs_loss"]
        total     = l2 + self.lambda_reg * sigreg
        return {"total": total, "l2": l2, "sigreg": sigreg}
