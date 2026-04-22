import torch
import pytest


def test_smooth_l1_decreases_toward_target():
    """smoothl1 component decreases as prediction approaches target."""
    from src.losses.prediction import TextJEPALoss
    loss_fn = TextJEPALoss(lambda_reg=0.0)
    target = torch.randn(4, 256)
    far  = torch.randn(4, 256)
    near = target + torch.randn(4, 256) * 0.01
    loss_far  = loss_fn(far,  target)["total"]
    loss_near = loss_fn(near, target)["total"]
    assert loss_near < loss_far, "loss should be smaller when prediction is near target"


def test_sigreg_detects_collapsed_representations():
    """sigreg (bcs) loss is higher when all embeddings are identical (collapsed)."""
    from src.losses.prediction import TextJEPALoss
    loss_fn = TextJEPALoss(lambda_reg=1.0)
    target    = torch.randn(8, 256)
    collapsed = torch.zeros(8, 256)
    diverse   = torch.randn(8, 256)
    loss_collapsed = loss_fn(collapsed, target)["sigreg"]
    loss_diverse   = loss_fn(diverse,   target)["sigreg"]
    assert loss_collapsed > loss_diverse, \
        "sigreg should penalize collapsed representations more"


def test_no_cross_entropy_in_training_path():
    """textjepaloss contains no cross-entropy loss term, verified by checking for CrossEntropyLoss and NLLLoss modules."""
    from src.losses.prediction import TextJEPALoss
    import torch.nn as nn
    loss_fn = TextJEPALoss(lambda_reg=1.0)
    ce_modules = [
        n for n, m in loss_fn.named_modules()
        if isinstance(m, (nn.CrossEntropyLoss, nn.NLLLoss))
    ]
    assert len(ce_modules) == 0, f"cross-entropy found in loss: {ce_modules}"


def test_loss_uses_bcs_class_from_anticollapse():
    """loss function uses BCS from src.losses.anticollapse, copied from eb_jepa, not reimplemented."""
    from src.losses.prediction import TextJEPALoss
    from src.losses.anticollapse import BCS
    loss_fn = TextJEPALoss(lambda_reg=1.0)
    bcs_modules = [m for m in loss_fn.modules() if isinstance(m, BCS)]
    assert len(bcs_modules) == 1, \
        f"expected exactly 1 BCS module in loss, got {len(bcs_modules)}"


def test_loss_stop_grad_on_target():
    """calling backward through total loss must not produce gradients on the target tensor (stop-gradient invariant)."""
    from src.losses.prediction import TextJEPALoss
    loss_fn = TextJEPALoss(lambda_reg=0.0)
    pred   = torch.randn(4, 256, requires_grad=True)
    target = torch.randn(4, 256, requires_grad=True)
    out = loss_fn(pred, target)
    out["total"].backward()
    assert target.grad is None or torch.all(target.grad == 0), \
        "gradient flowed to target tensor; stop-grad not applied"


def test_bcs_defaults_match_hier_cost_exp():
    """bcs is initialized with num_slices=1024 and lmbd=0.1, matching hier_cost_exp branch defaults."""
    from src.losses.prediction import TextJEPALoss
    from src.losses.anticollapse import BCS
    loss_fn = TextJEPALoss(lambda_reg=1.0)
    bcs = next(m for m in loss_fn.modules() if isinstance(m, BCS))
    assert bcs.num_slices == 1024, f"expected num_slices=1024, got {bcs.num_slices}"
    assert bcs.lmbd == 0.1, f"expected lmbd=0.1, got {bcs.lmbd}"
