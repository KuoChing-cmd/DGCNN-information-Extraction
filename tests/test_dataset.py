"""Unit tests for dataset and preprocessing."""

import unittest
import torch

from dgcnn_re.dataset import Vocab, Schema, RelationExtractionDataset, collate_fn


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.texts = ["北京是中国的首都", "上海是经济中心"]
        self.vocab = Vocab.build_from_texts(self.texts, min_count=1)
        self.schema = Schema.from_predicates(["首都", "中心"])

        self.sample_data = [
            {
                "text": "北京是中国的首都",
                "spo_list": [["中国", "首都", "北京"]],
            },
            {
                "text": "上海是经济中心",
                "spo_list": [["上海", "中心", "经济"]],
            },
        ]

    def test_vocab_encoding_decoding(self):
        ids = self.vocab.encode("北京")
        self.assertEqual(len(ids), 2)
        # Verify roundtrip
        decoded = self.vocab.decode(ids)
        self.assertEqual(decoded, "北京")

    def test_schema(self):
        self.assertEqual(len(self.schema), 2)
        self.assertIn("首都", self.schema.predicate2id)
        pred_id = self.schema.predicate2id["首都"]
        self.assertEqual(self.schema.id2predicate[pred_id], "首都")

    def test_dataset_item(self):
        dataset = RelationExtractionDataset(
            data=self.sample_data,
            vocab=self.vocab,
            schema=self.schema,
            max_len=64,
            is_training=True,
        )
        self.assertEqual(len(dataset), 2)
        item = dataset[0]
        self.assertIn("char_ids", item)
        self.assertIn("s1", item)
        self.assertIn("s2", item)
        self.assertIn("o1", item)
        self.assertIn("o2", item)
        self.assertEqual(len(item["char_ids"]), len(item["text"]))

    def test_collate_fn(self):
        dataset = RelationExtractionDataset(
            data=self.sample_data,
            vocab=self.vocab,
            schema=self.schema,
            max_len=64,
            is_training=True,
        )
        batch = [dataset[0], dataset[1]]
        collated = collate_fn(batch)

        self.assertIn("char_ids", collated)
        self.assertIn("mask", collated)
        self.assertIn("s1", collated)
        self.assertIn("k1", collated)
        self.assertEqual(collated["char_ids"].dim(), 2)
        self.assertEqual(collated["s1"].dim(), 2)
        self.assertEqual(collated["o1"].dim(), 3)


if __name__ == "__main__":
    unittest.main()
