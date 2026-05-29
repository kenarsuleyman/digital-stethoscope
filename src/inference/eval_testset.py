"""
Evaluate the model on OUR OWN precomputed mel splits (the exact arrays used in
training/evaluation). This isolates the MODEL from any preprocessing question:
the .npy files are fed verbatim, so the result answers a single question --

  Does the model correctly classify its own held-out data?

  - High accuracy here + failure on external data  -> training is fine, the
    problem is dataset diversity / domain shift (need more varied data).
  - Low accuracy here too                           -> the training itself is
    broken (architecture, labels, preprocessing, or the conversion to tflite).

Usage:
    python -m src.inference.eval_testset --model PATH/heart_sound_classifier_fp16.tflite
    python -m src.inference.eval_testset --model model.tflite --split splits/val.csv
    python -m src.inference.eval_testset --model model.tflite --limit 300   # quick
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from src.inference.predict_wav import load_backend

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def evaluate(model_path: Path, split_csv: Path, threshold: float, limit: int | None):
    with open(split_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(rows))[:limit]
        rows = [rows[i] for i in sorted(idx)]

    backend = load_backend(model_path)

    y_true = np.array([int(r["binary_label"]) for r in rows])
    probs = np.empty(len(rows), dtype=np.float64)

    for i, r in enumerate(rows):
        mel = np.load(PROJECT_ROOT / r["mel_path"]).astype(np.float32)
        probs[i] = backend.predict_batch([mel])[0]
        if (i + 1) % 200 == 0:
            logger.info(f"  {i + 1}/{len(rows)} done")

    y_pred = (probs >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = len(y_true)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    auc = _roc_auc(y_true, probs)

    print("\n" + "=" * 62)
    print(f"  SPLIT: {split_csv.name}   ({n} segments)   threshold={threshold}")
    print(f"  Normal: {(y_true == 0).sum()}   Abnormal: {(y_true == 1).sum()}")
    print("-" * 62)
    print("  CONFUSION MATRIX               pred:Normal  pred:Abnormal")
    print(f"  truth:Normal                   {tn:>10}   {fp:>12}")
    print(f"  truth:Abnormal                 {fn:>10}   {tp:>12}")
    print("-" * 62)
    print(f"  Accuracy:    {acc:.4f}   ({tp + tn}/{n})")
    print(f"  Sensitivity: {sens:.4f}   (abnormal recall)")
    print(f"  Specificity: {spec:.4f}   (normal recall)")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Prob stats:  normal mean={probs[y_true == 0].mean():.3f}  "
          f"abnormal mean={probs[y_true == 1].mean():.3f}")
    print("=" * 62)


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUC, no sklearn dependency."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    rank_pos = ranks[y_true == 1].sum()
    return (rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    p = argparse.ArgumentParser(description="Evaluate model on own mel splits")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--split", type=Path, default=PROJECT_ROOT / "splits" / "test.csv")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None, help="Random subset for a quick run")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    evaluate(args.model, args.split, args.threshold, args.limit)


if __name__ == "__main__":
    main()
