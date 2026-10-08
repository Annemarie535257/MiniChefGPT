"""Causal self-attention used by the recipe Transformer.

The implementation uses NumPy so it can be inspected and tested without a
framework-specific training loop. Tensors use batch-first shapes:

    inputs:  (batch, sequence_length, embedding_dim)
    output:  (batch, sequence_length, embedding_dim)

For a token at position ``i``, the causal mask prevents attention to every
position ``j > i``. This is the property needed for next-token prediction:
the representation at a position cannot see the token it is supposed to
predict.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


Array = np.ndarray


def causal_mask(sequence_length: int) -> Array:
    """Return a boolean mask where ``True`` marks permitted attention links."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    return np.tril(np.ones((sequence_length, sequence_length), dtype=bool))


def _softmax(values: Array, axis: int = -1) -> Array:
    """Compute a numerically stable softmax along one axis."""
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Calculate attention outputs and weights.

    Query asks which information the current token needs. Key describes what
    each token contains, and Value is the information that is retrieved when a
    key is selected. The dot product compares a query with every key; scaling
    keeps logits numerically stable; softmax turns them into weights.

    Args:
        query: ``(..., query_length, depth)``.
        key: ``(..., key_length, depth)``.
        value: ``(..., key_length, value_depth)``.
        mask: Boolean array broadcastable to ``(..., query_length, key_length)``.
            ``True`` means attention is allowed.

    Returns:
        A tuple ``(output, weights)``.
    """
    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("query, key, and value must have at least 2 dimensions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key depth must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value sequence lengths must match")

    depth = query.shape[-1]
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(depth)
    if mask is not None:
        allowed = np.asarray(mask, dtype=bool)
        try:
            np.broadcast_to(allowed, scores.shape)
        except ValueError as error:
            raise ValueError(
                f"mask shape {allowed.shape} is not broadcastable to scores "
                f"shape {scores.shape}"
            ) from error
        scores = np.where(allowed, scores, -np.inf)

    weights = _softmax(scores, axis=-1)
    return np.matmul(weights, value), weights


class MultiHeadSelfAttention:
    """Multi-head causal self-attention with batch-first inputs."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        *,
        seed: Optional[int] = None,
    ) -> None:
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        generator = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(embedding_dim)
        self.query_projection = generator.normal(
            0.0, scale, (embedding_dim, embedding_dim)
        )
        self.key_projection = generator.normal(
            0.0, scale, (embedding_dim, embedding_dim)
        )
        self.value_projection = generator.normal(
            0.0, scale, (embedding_dim, embedding_dim)
        )
        self.output_projection = generator.normal(
            0.0, scale, (embedding_dim, embedding_dim)
        )

    def _project(self, inputs: Array, projection: Array) -> Array:
        projected = np.matmul(inputs, projection)
        batch_size, sequence_length, _ = projected.shape
        return projected.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

    def forward(
        self,
        inputs: Array,
        attention_mask: Optional[Array] = None,
        *,
        return_attention: bool = False,
    ) -> Array | Tuple[Array, Array]:
        """Apply causal multi-head attention.

        ``attention_mask`` can further block positions, but cannot disable the
        causal mask. It may have shape ``(sequence, sequence)`` or
        ``(batch, sequence, sequence)``.
        """
        values = np.asarray(inputs, dtype=float)
        if values.ndim != 3:
            raise ValueError(
                "inputs must have shape (batch, sequence_length, embedding_dim)"
            )
        if values.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"last input dimension must be {self.embedding_dim}, "
                f"got {values.shape[-1]}"
            )

        batch_size, sequence_length, _ = values.shape
        query = self._project(values, self.query_projection)
        key = self._project(values, self.key_projection)
        value = self._project(values, self.value_projection)

        mask = causal_mask(sequence_length)
        if attention_mask is not None:
            extra_mask = np.asarray(attention_mask, dtype=bool)
            try:
                mask = np.logical_and(mask, extra_mask)
                np.broadcast_to(mask, (batch_size, sequence_length, sequence_length))
            except ValueError as error:
                raise ValueError(
                    "attention_mask must be shaped (sequence, sequence) or "
                    "(batch, sequence, sequence)"
                ) from error

        attended, weights = scaled_dot_product_attention(query, key, value, mask)
        combined = attended.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.embedding_dim
        )
        output = np.matmul(combined, self.output_projection)
        if return_attention:
            return output, weights
        return output

    __call__ = forward
