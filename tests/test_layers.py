"""Unit tests for neural network layers."""

import unittest
import torch

from dgcnn_re.layers import (
    PositionalEncoding,
    DilatedGatedConv1d,
    MultiHeadSelfAttention,
    MaskedGlobalMaxPool1d,
)


class TestLayers(unittest.TestCase):
    def test_positional_encoding(self):
        pe = PositionalEncoding(d_model=64, max_len=128, learnable=False)
        pos = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        out = pe(pos)
        self.assertEqual(out.shape, (2, 4, 64))

        # Learnable
        pe_learnable = PositionalEncoding(d_model=64, max_len=128, learnable=True)
        out_learnable = pe_learnable(pos)
        self.assertEqual(out_learnable.shape, (2, 4, 64))

    def test_dilated_gated_conv1d(self):
        batch_size, seq_len, channels = 2, 16, 32
        x = torch.randn(batch_size, seq_len, channels)
        mask = torch.ones(batch_size, seq_len)
        mask[:, 12:] = 0.0

        conv = DilatedGatedConv1d(channels=channels, kernel_size=3, dilation=2)
        out = conv(x, mask=mask)

        # Output shape must preserve sequence length and channels
        self.assertEqual(out.shape, (batch_size, seq_len, channels))
        # Masked region should be zero
        self.assertTrue(torch.all(out[:, 12:, :] == 0))

    def test_multi_head_attention(self):
        batch_size, seq_len, d_model = 2, 10, 64
        q = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, seq_len)
        mask[:, 8:] = 0.0

        attn = MultiHeadSelfAttention(
            d_model=d_model, num_heads=4, size_per_head=16
        )
        out = attn(q=q, key_mask=mask, query_mask=mask)

        self.assertEqual(out.shape, (batch_size, seq_len, 4 * 16))
        # Masked query positions must be zero
        self.assertTrue(torch.all(out[:, 8:, :] == 0))

    def test_masked_max_pooling(self):
        x = torch.tensor(
            [
                [[1.0, 2.0], [5.0, 3.0], [10.0, 10.0]],
            ]
        )  # (1, 3, 2)
        mask = torch.tensor([[1.0, 1.0, 0.0]])  # last element masked

        pool = MaskedGlobalMaxPool1d()
        pooled = pool(x, mask)
        # Should be max of first 2 positions -> [5.0, 3.0]
        self.assertEqual(pooled.shape, (1, 2))
        self.assertEqual(pooled[0, 0].item(), 5.0)
        self.assertEqual(pooled[0, 1].item(), 3.0)


if __name__ == "__main__":
    unittest.main()
