import torch.nn as nn


class LinearProbe(nn.Module):
    """linear probe for downstream eval. trained separately on frozen encoder."""
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_classes)

    def forward(self, z):
        return self.linear(z)
