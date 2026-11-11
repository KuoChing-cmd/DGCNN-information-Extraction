#!/usr/bin/env python3
"""Training entrypoint for DGCNN Relation Extraction."""

import argparse
import json
import logging
import os
import random
import numpy as np
import torch

from dgcnn_re.config import Config
from dgcnn_re.dataset import Vocab, Schema, RelationExtractionDataset
from dgcnn_re.model import DGCNNForRelationExtraction
from dgcnn_re.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train DGCNN Relation Extraction Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML configuration")
    parser.add_argument("--train_path", type=str, default=None, help="Path to training data JSON")
    parser.add_argument("--val_path", type=str, default=None, help="Path to validation data JSON")
    parser.add_argument("--schema_path", type=str, default=None, help="Path to schema JSON")
    parser.add_argument("--vocab_path", type=str, default=None, help="Path to vocabulary JSON")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Compute device (auto, cuda, mps, cpu)")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save checkpoints")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load configuration
    if os.path.exists(args.config):
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config()

    # CLI Overrides
    if args.train_path:
        cfg.data.train_path = args.train_path
    if args.val_path:
        cfg.data.val_path = args.val_path
    if args.schema_path:
        cfg.data.schema_path = args.schema_path
    if args.vocab_path:
        cfg.data.vocab_path = args.vocab_path
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.lr:
        cfg.training.lr = args.lr
    if args.device:
        cfg.training.device = args.device
    if args.save_dir:
        cfg.training.save_dir = args.save_dir

    set_seed(cfg.training.seed)

    # Load raw data
    if not os.path.exists(cfg.data.train_path):
        raise FileNotFoundError(f"Training dataset not found: {cfg.data.train_path}")
    with open(cfg.data.train_path, "r", encoding="utf-8") as f:
        train_raw = json.load(f)

    if cfg.data.val_path and os.path.exists(cfg.data.val_path):
        with open(cfg.data.val_path, "r", encoding="utf-8") as f:
            val_raw = json.load(f)
    else:
        logging.info("No separate validation set specified; splitting 15% from train data.")
        indices = list(range(len(train_raw)))
        random.shuffle(indices)
        split = max(1, int(len(train_raw) * 0.15))
        val_raw = [train_raw[i] for i in indices[:split]]
        train_raw = [train_raw[i] for i in indices[split:]]

    # Load or build Vocab & Schema
    if os.path.exists(cfg.data.vocab_path):
        vocab = Vocab.load(cfg.data.vocab_path)
    else:
        logging.info("Building vocabulary from training data...")
        vocab = Vocab.build_from_texts([d["text"] for d in train_raw], min_count=cfg.data.min_char_count)
        vocab.save(cfg.data.vocab_path)

    if os.path.exists(cfg.data.schema_path):
        schema = Schema.load(cfg.data.schema_path)
    else:
        logging.info("Building relation schema from training data...")
        preds = set()
        for d in train_raw:
            for sp in d.get("spo_list", []):
                p = sp[1] if isinstance(sp, (list, tuple)) else sp["predicate"]
                preds.add(p)
        schema = Schema.from_predicates(list(preds))
        schema.save(cfg.data.schema_path)

    cfg.model.vocab_size = len(vocab)
    cfg.model.num_classes = len(schema)

    logging.info(f"Vocabulary size: {len(vocab)} | Relation classes: {len(schema)}")

    train_dataset = RelationExtractionDataset(
        data=train_raw,
        vocab=vocab,
        schema=schema,
        max_len=cfg.model.max_len,
        is_training=True,
    )
    val_dataset = RelationExtractionDataset(
        data=val_raw,
        vocab=vocab,
        schema=schema,
        max_len=cfg.model.max_len,
        is_training=False,
    )

    model = DGCNNForRelationExtraction(cfg.model)
    trainer = Trainer(
        config=cfg,
        model=model,
        vocab=vocab,
        schema=schema,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()


if __name__ == "__main__":
    main()

