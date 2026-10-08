"""
residual.py - Residual (skip) connections for MiniChefGPT.
Owner: Marie Claire

A Transformer block uses TWO residual connections:
    1st residual -> around masked multi-head attention
    2nd residual -> around the feed-forward network (FFN)

Usage inside TransformerBlock (pre-LayerNorm style):
    self.residual1 = ResidualConnection(dropout)
    self.residual2 = ResidualConnection(dropout)

    def forward(self, x):
        x = self.residual1(x, self.attention(self.ln1(x)))
        x = self.residual2(x, self.ffn(self.ln2(x)))
        return x

Tensor shapes: (batch_size, seq_len, d_model)
"""

import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    """output = x + dropout(sublayer_output)"""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        if x.shape != sublayer_output.shape:
            raise ValueError(
                f"Residual shape mismatch: x {tuple(x.shape)} vs "
                f"sublayer_output {tuple(sublayer_output.shape)}"
            )
        return x + self.dropout(sublayer_output)


if __name__ == "__main__":
    B, T, D = 2, 16, 64
    x = torch.randn(B, T, D)
    out = ResidualConnection(0.1)(x, torch.randn(B, T, D))
    assert out.shape == (B, T, D)
    print("residual.py OK, output shape:", tuple(out.shape))