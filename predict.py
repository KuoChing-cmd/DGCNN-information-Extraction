#!/usr/bin/env python3
"""Prediction and inference CLI for DGCNN Relation Extraction."""

import argparse
import json
import os
import sys
import torch

from dgcnn_re.predictor import RelationExtractor


def main():
    parser = argparse.ArgumentParser(description="Predict relations using trained DGCNN-RE model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--vocab", type=str, default="data/vocab.json", help="Path to vocab JSON")
    parser.add_argument("--schema", type=str, default="data/sample_schemas.json", help="Path to schema JSON")
    parser.add_argument("--text", type=str, default=None, help="Input sentence to extract")
    parser.add_argument("--input_file", type=str, default=None, help="Path to JSON file with input samples")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save prediction results")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, mps, cpu)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found at '{args.checkpoint}'. Please train the model first.")
        sys.exit(1)

    device = None
    if args.device != "auto":
        device = torch.device(args.device)

    print("Loading model and artifacts...")
    extractor = RelationExtractor.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        schema_path=args.schema,
        device=device,
    )
    print("Ready!")

    if args.text:
        res = extractor.extract(args.text)
        print("\nInput text:", args.text)
        print("Extracted SPO Triplets:")
        for triplet in res["triplets"]:
            print(f"  - Subject: [{triplet[0]}]  Predicate: [{triplet[1]}]  Object: [{triplet[2]}]")
        return

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = []
        for item in data:
            text = item if isinstance(item, str) else item.get("text", "")
            res = extractor.extract(text)
            results.append(res)

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {args.output_file}")
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    # Interactive REPL
    print("\n=== DGCNN Relation Extraction Interactive Mode ===")
    print("Enter sentence to extract relations (or 'exit' / 'quit' to end):\n")
    try:
        while True:
            text = input("Text > ").strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit"):
                break
            res = extractor.extract(text)
            print(f"Detected {len(res['triplets'])} triplet(s):")
            for t in res["triplets"]:
                print(f"  ({t[0]}, {t[1]}, {t[2]})")
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")


if __name__ == "__main__":
    main()
