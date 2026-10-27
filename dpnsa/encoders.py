import torch.nn as nn


class ResNetTabularEncoder(nn.Module):
    """Residual MLP encoder used by every DP-NSA clone."""

    def __init__(self, d_in, d_lat=64, n_blocks=3, d_hidden=128, dropout=0.2):
        super().__init__()
        self.d_lat = d_lat

        self.input_proj = nn.Linear(d_in, d_hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.LayerNorm(d_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_hidden),
                nn.LayerNorm(d_hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(n_blocks)
        ])
        self.head = nn.Linear(d_hidden, d_lat)

    def forward(self, x):
        z = self.input_proj(x)
        for block in self.blocks:
            z = z + block(z)
        return self.head(z)
