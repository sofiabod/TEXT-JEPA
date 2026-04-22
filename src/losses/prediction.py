import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.anticollapse import BCS


class TextJEPALoss(nn.Module):
    """l = smoothl1(z_pred, sg(z_target)) + lambda_reg * l_bcs. z_target is detached before any computation. bcs imported from src.losses.anticollapse, copied verbatim from eb_jepa hier_cost_exp."""
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
        """z_pred and z_target are (B, d); z_target is detached. returns dict with total, smooth_l1, sigreg."""
        z_target_sg = z_target.detach()
        smooth_l1 = F.smooth_l1_loss(z_pred, z_target_sg)
        bcs_out   = self.bcs(z_pred, z_target_sg)
        sigreg    = bcs_out["bcs_loss"]
        total     = smooth_l1 + self.lambda_reg * sigreg
        return {"total": total, "smooth_l1": smooth_l1, "sigreg": sigreg}
