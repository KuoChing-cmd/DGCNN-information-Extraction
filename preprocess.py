#!/usr/bin/env python3
"""Data preprocessing and dataset management script for DGCNN-RE."""

import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List, Any

from dgcnn_re.dataset import Vocab, Schema


def analyze_dataset(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    pred_counter = Counter()
    lengths = []
    num_spos = 0

    for item in data:
        text = item.get("text", "")
        lengths.append(len(text))
        for sp in item.get("spo_list", []):
            p = sp[1] if isinstance(sp, (list, tuple)) else sp.get("predicate", "")
            pred_counter[p] += 1
            num_spos += 1

    return {
        "num_samples": len(data),
        "num_triplets": num_spos,
        "avg_length": sum(lengths) / max(len(lengths), 1),
        "max_length": max(lengths) if lengths else 0,
        "predicate_counts": dict(pred_counter.most_common(20)),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess and prepare datasets for DGCNN-RE")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input raw JSON file")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for processed files")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--min_count", type=int, default=2, help="Minimum character frequency for vocabulary")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading raw data from: {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        # Support both json lines and single json array
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            raw_data = json.load(f)
        else:
            raw_data = [json.loads(line) for line in f if line.strip()]

    # Normalize samples
    cleaned_data = []
    for item in raw_data:
        text = item.get("text", "")
        spo_list = item.get("spo_list", [])
        norm_spo = []
        for sp in spo_list:
            if isinstance(sp, (list, tuple)) and len(sp) == 3:
                norm_spo.append([sp[0], sp[1], sp[2]])
            elif isinstance(sp, dict):
                norm_spo.append([sp["subject"], sp["predicate"], sp["object"]])
        if text and norm_spo:
            cleaned_data.append({"text": text, "spo_list": norm_spo})

    print(f"Valid samples with relations: {len(cleaned_data)}")

    # Shuffle and split
    random.shuffle(cleaned_data)
    val_size = int(len(cleaned_data) * args.val_ratio)
    val_data = cleaned_data[:val_size]
    train_data = cleaned_data[val_size:]

    # Save splits
    train_out = os.path.join(args.output_dir, "train.json")
    val_out = os.path.join(args.output_dir, "val.json")
    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    # Build Vocab
    vocab = Vocab.build_from_texts([d["text"] for d in train_data], min_count=args.min_count)
    vocab_out = os.path.join(args.output_dir, "vocab.json")
    vocab.save(vocab_out)

    # Build Schema
    preds = set()
    for d in cleaned_data:
        for sp in d["spo_list"]:
            preds.add(sp[1])
    schema = Schema.from_predicates(list(preds))
    schema_out = os.path.join(args.output_dir, "schemas.json")
    schema.save(schema_out)

    # Report statistics
    stats = analyze_dataset(cleaned_data)
    print("\nDataset Statistics:")
    print(f"  Total samples: {stats['num_samples']}")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Relation classes: {len(schema)}")
    print(f"  Total triplets: {stats['num_triplets']}")
    print(f"  Average text length: {stats['avg_length']:.1f}")
    print("\nTop Predicates:")
    for pred, count in stats["predicate_counts"].items():
        print(f"    - {pred}: {count}")


if __name__ == "__main__":
    main()
