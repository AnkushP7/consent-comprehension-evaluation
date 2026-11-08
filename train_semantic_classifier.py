"""Supervised TF-IDF baseline and NLP classifier trainer.

Usage:
  python train_semantic_classifier.py [path_to_csv] [--output models/semantic_review_model.joblib]

If no CSV path is provided, it defaults to the curated dataset:
  data/annotated_consent_sentences.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nlp_classifier import train_classifier

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "annotated_consent_sentences.csv"
DEFAULT_OUTPUT = ROOT / "models" / "semantic_review_model.joblib"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_DATASET),
        help=f"Annotated CSV with text and label columns (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output model path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"Dataset file not found: {csv_path}")

    print(f"Training NLP Consent Classifier on: {csv_path}")
    metrics = train_classifier(
        csv_path=csv_path,
        output_path=args.output,
        test_size=args.test_size,
        seed=args.seed,
    )

    print("\n" + "=" * 50)
    print("TRAINING AND EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model Type:      {metrics['model_type']}")
    print(f"Total Rows:      {metrics['rows_total']} (Train: {metrics['rows_train']}, Test: {metrics['rows_test']})")
    print(f"Split Method:    {metrics['split_method']}")
    print(f"Accuracy:        {metrics['accuracy']:.2%}")
    print(f"Macro-F1:        {metrics['macro_f1']:.4f}")
    print(f"Macro-Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro-Recall:    {metrics['macro_recall']:.4f}")
    print(f"Classes:         {metrics['classes']}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    print("-" * 50)
    print(f"Model artifact saved to:   {args.output}")
    print(f"Metrics artifact saved to: {Path(args.output).with_suffix('.metrics.json')}")
    print("=" * 50)

    top_rev = metrics.get("top_features", {}).get("needs_review", [])
    if top_rev:
        print("\nTop n-grams predictive of 'needs_review':")
        for f in top_rev[:8]:
            print(f"  + {f['feature']} (weight: {f['weight']})")

    top_clr = metrics.get("top_features", {}).get("clear", [])
    if top_clr:
        print("\nTop n-grams predictive of 'clear':")
        for f in top_clr[:8]:
            print(f"  - {f['feature']} (weight: {f['weight']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
