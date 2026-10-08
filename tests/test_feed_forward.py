import unittest

import torch

from src.feed_forward import FeedForward


class FeedForwardTests(unittest.TestCase):
    def test_preserves_batch_and_sequence_dimensions(self) -> None:
        torch.manual_seed(0)
        network = FeedForward(d_model=16, d_ff=64)
        inputs = torch.randn(2, 7, 16)

        outputs = network(inputs)

        self.assertEqual(outputs.shape, inputs.shape)
        self.assertFalse(torch.equal(outputs, inputs))

    def test_applies_the_same_network_independently_to_each_token(self) -> None:
        torch.manual_seed(1)
        network = FeedForward(d_model=8, d_ff=32)
        inputs = torch.randn(2, 5, 8)

        batched_outputs = network(inputs)
        individual_outputs = torch.stack(
            [network(inputs[:, position, :]) for position in range(inputs.size(1))],
            dim=1,
        )

        torch.testing.assert_close(batched_outputs, individual_outputs)

    def test_rejects_an_incompatible_embedding_dimension(self) -> None:
        network = FeedForward(d_model=8, d_ff=32)

        with self.assertRaisesRegex(ValueError, "last input dimension"):
            network(torch.randn(2, 5, 10))


if __name__ == "__main__":
    unittest.main()
