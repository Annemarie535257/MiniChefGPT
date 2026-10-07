import unittest

import numpy as np

from src.attention import (
    MultiHeadSelfAttention,
    causal_mask,
    scaled_dot_product_attention,
)


class CausalAttentionTests(unittest.TestCase):
    def test_causal_mask_hides_future_positions(self):
        mask = causal_mask(4)
        np.testing.assert_array_equal(
            mask,
            np.array(
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, False],
                    [True, True, True, True],
                ]
            ),
        )

    def test_scaled_attention_assigns_zero_weight_to_future_tokens(self):
        values = np.ones((1, 1, 4, 2))
        _, weights = scaled_dot_product_attention(values, values, values, causal_mask(4))
        self.assertEqual(weights.shape, (1, 1, 4, 4))
        self.assertTrue(np.all(weights[0, 0][~causal_mask(4)] == 0.0))
        np.testing.assert_allclose(weights[0, 0].sum(axis=-1), 1.0)

    def test_future_tokens_do_not_change_earlier_outputs(self):
        attention = MultiHeadSelfAttention(embedding_dim=4, num_heads=2, seed=7)
        prefix = np.arange(12, dtype=float).reshape(1, 3, 4)
        first = np.concatenate([prefix, np.zeros((1, 2, 4))], axis=1)
        second = np.concatenate([prefix, np.full((1, 2, 4), 1000.0)], axis=1)

        first_output = attention(first)
        second_output = attention(second)
        np.testing.assert_allclose(first_output[:, :3], second_output[:, :3])

    def test_multi_head_shapes_and_attention_shape(self):
        attention = MultiHeadSelfAttention(embedding_dim=8, num_heads=4, seed=3)
        output, weights = attention(np.ones((2, 5, 8)), return_attention=True)
        self.assertEqual(output.shape, (2, 5, 8))
        self.assertEqual(weights.shape, (2, 4, 5, 5))

    def test_invalid_head_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(embedding_dim=5, num_heads=2)


if __name__ == "__main__":
    unittest.main()
