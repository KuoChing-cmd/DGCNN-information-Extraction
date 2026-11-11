"""Unit tests for DGCNN Relation Extraction model architecture and loss."""

import unittest
import torch

from dgcnn_re.config import ModelConfig
from dgcnn_re.model import DGCNNForRelationExtraction
from dgcnn_re.loss import MaskedRelationExtractionLoss


class TestModel(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            vocab_size=100,
            num_classes=5,
            char_dim=32,
            hidden_dim=32,
            max_len=64,
            dilation_rates=[1, 2, 1],
            num_heads=2,
            size_per_head=16,
            dropout=0.1,
            conv_dropout=0.1,
        )
        self.model = DGCNNForRelationExtraction(self.config)
        self.loss_fn = MaskedRelationExtractionLoss()

    def test_forward_pass_training(self):
        batch_size, seq_len = 2, 16
        char_ids = torch.randint(1, 99, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)
        mask[:, 12:] = 0.0

        k1 = torch.tensor([[2], [4]], dtype=torch.long)
        k2 = torch.tensor([[4], [6]], dtype=torch.long)

        out = self.model(char_ids=char_ids, k1=k1, k2=k2, mask=mask)

        self.assertIn("ps1", out)
        self.assertIn("ps2", out)
        self.assertIn("po1", out)
        self.assertIn("po2", out)

        self.assertEqual(out["ps1"].shape, (batch_size, seq_len, 1))
        self.assertEqual(out["ps2"].shape, (batch_size, seq_len, 1))
        self.assertEqual(out["po1"].shape, (batch_size, seq_len, self.config.num_classes))
        self.assertEqual(out["po2"].shape, (batch_size, seq_len, self.config.num_classes))

    def test_loss_and_backward(self):
        batch_size, seq_len = 2, 16
        char_ids = torch.randint(1, 99, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)
        k1 = torch.tensor([[1], [2]], dtype=torch.long)
        k2 = torch.tensor([[3], [4]], dtype=torch.long)

        out = self.model(char_ids=char_ids, k1=k1, k2=k2, mask=mask)

        # Mock ground truths
        s1 = torch.zeros(batch_size, seq_len)
        s2 = torch.zeros(batch_size, seq_len)
        s1[:, 1] = 1.0
        s2[:, 3] = 1.0

        o1 = torch.zeros(batch_size, seq_len, self.config.num_classes)
        o2 = torch.zeros(batch_size, seq_len, self.config.num_classes)
        o1[:, 5, 0] = 1.0
        o2[:, 7, 0] = 1.0

        loss, loss_dict = self.loss_fn(
            ps1=out["ps1"],
            ps2=out["ps2"],
            po1=out["po1"],
            po2=out["po2"],
            s1=s1,
            s2=s2,
            o1=o1,
            o2=o2,
            mask=mask,
        )

        self.assertGreater(loss.item(), 0.0)
        self.assertIn("loss_total", loss_dict)
        self.assertIn("loss_s1", loss_dict)
        self.assertIn("loss_o1", loss_dict)

        # Verify gradient backprop
        self.model.zero_grad()
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None!")


if __name__ == "__main__":
    unittest.main()
