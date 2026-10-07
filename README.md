# MiniChefGPT
The transformer model that will help us to get recipe from the ingredients we have.

## Task 2: masked attention

The attention layer is in [`src/attention.py`](src/attention.py). It uses
NumPy arrays with shape `(batch, sequence_length, embedding_dim)`.
For each token, attention compares:

- **Query (Q):** what information the current token is looking for.
- **Key (K):** what information each available token contains.
- **Value (V):** the content retrieved when a key is selected.

The attention score is `softmax(QK^T / sqrt(depth))V`. The lower-triangular
causal mask forces every position `i` to use only positions `0..i`, hiding
future recipe tokens during next-token prediction. An optional extra mask can
hide padding or other positions, but it can never enable a future position.

Example:

```python
import numpy as np
from src.attention import MultiHeadSelfAttention

layer = MultiHeadSelfAttention(embedding_dim=128, num_heads=8, seed=42)
embeddings = np.zeros((batch_size, sequence_length, 128))
output, weights = layer(embeddings, return_attention=True)
# output: (batch_size, sequence_length, 128)
# weights: (batch_size, 8, sequence_length, sequence_length)
```

Run the focused tests from the repository root with:

```text
python -m unittest discover -s tests -v
```
