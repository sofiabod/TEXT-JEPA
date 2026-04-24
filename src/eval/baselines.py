import torch


def copy_forward_predictions(z_context: torch.Tensor) -> torch.Tensor:
    """copy-forward baseline: predict next segment = last context segment embedding."""
    return z_context[:, -1, :]


def zero_baseline_predictions(z_context: torch.Tensor) -> torch.Tensor:
    """zero vector baseline, should be the worst performer."""
    return torch.zeros(
        z_context.shape[0], z_context.shape[2],
        device=z_context.device, dtype=z_context.dtype,
    )
