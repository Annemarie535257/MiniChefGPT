"""Position-wise feed-forward network used in Transformer blocks."""

from torch import Tensor, nn


class FeedForward(nn.Module):
    """Apply a two-layer MLP independently to every token representation."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive integers")

        self.d_model = d_model
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Transform the last dimension, preserving all leading dimensions."""
        if x.ndim < 2:
            raise ValueError("input must include a sequence or batch dimension")
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"expected the last input dimension to be {self.d_model}, "
                f"got {x.size(-1)}"
            )
        return self.network(x)
