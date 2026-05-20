"""
Test Set Evaluation

Evaluates the trained model on the test set and produces:
- Metrics: accuracy, sensitivity, specificity, precision, F1, AUC-ROC
- Confusion matrix figure
- ROC curve figure
- Precision-Recall curve figure
- Results JSON and markdown report

Usage (on Colab):
    python -m src.evaluation.evaluate --data-dir /content/data --model-path /content/output/models/best_model.keras --output-dir /content/output
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score,
)

logger = logging.getLogger(__name__)


def load_test_data(split_csv: Path, project_root: Path):
    """Load test set .npy paths and labels."""
    with open(split_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    paths = [str(project_root / r["mel_path"]) for r in rows]
    labels = np.array([int(r["binary_label"]) for r in rows])
    return paths, labels


def predict_test_set(model, paths: list, batch_size: int = 32) -> np.ndarray:
    """Run inference on all test samples, return prediction probabilities."""
    predictions = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch = np.stack([np.load(p) for p in batch_paths])
        preds = model.predict(batch, verbose=0)
        predictions.append(preds.flatten())
    return np.concatenate(predictions)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Compute all classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity_recall": float(recall_score(y_true, y_pred)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "total_samples": int(len(y_true)),
        "positive_samples": int(np.sum(y_true)),
        "negative_samples": int(np.sum(1 - y_true)),
    }


def generate_figures(y_true, y_prob, metrics, output_dir: Path):
    """Generate all evaluation figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["font.family"] = "DejaVu Sans"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    y_pred = (y_prob >= 0.5).astype(int)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Anormal"],
                yticklabels=["Normal", "Anormal"],
                cbar_kws={"label": "Sayı"})
    # Add percentages
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, f"(%{cm_pct[i, j]:.1f})",
                    ha="center", va="center", fontsize=10, color="gray")

    ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=12)
    ax.set_ylabel("Gerçek Sınıf", fontsize=12)
    ax.set_title("Karışıklık Matrisi", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = metrics["auc_roc"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Rastgele sınıflandırıcı")
    ax.set_xlabel("Yanlış Pozitif Oranı (FPR)", fontsize=12)
    ax.set_ylabel("Doğru Pozitif Oranı (TPR)", fontsize=12)
    ax.set_title("ROC Eğrisi", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Precision-Recall Curve ---
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = metrics["average_precision"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec, prec, "darkorange", linewidth=2, label=f"PR (AP = {ap:.4f})")
    ax.set_xlabel("Duyarlılık (Recall)", fontsize=12)
    ax.set_ylabel("Kesinlik (Precision)", fontsize=12)
    ax.set_title("Kesinlik-Duyarlılık Eğrisi", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved: fig_confusion_matrix.png, fig_roc_curve.png, fig_pr_curve.png")


def save_results(metrics: dict, output_dir: Path):
    """Save metrics as JSON and markdown report."""
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Markdown report
    cm = metrics["confusion_matrix"]
    report = f"""# Test Seti Değerlendirme Sonuçları

| Metrik | Değer |
|--------|-------|
| Doğruluk (Accuracy) | {metrics['accuracy']:.4f} |
| Duyarlılık (Sensitivity/Recall) | {metrics['sensitivity_recall']:.4f} |
| Özgüllük (Specificity) | {metrics['specificity']:.4f} |
| Kesinlik (Precision) | {metrics['precision']:.4f} |
| F1 Skoru | {metrics['f1_score']:.4f} |
| AUC-ROC | {metrics['auc_roc']:.4f} |
| Ortalama Kesinlik (AP) | {metrics['average_precision']:.4f} |

## Karışıklık Matrisi

|  | Tahmin: Normal | Tahmin: Anormal |
|--|----------------|-----------------|
| Gerçek: Normal | {cm['tn']} | {cm['fp']} |
| Gerçek: Anormal | {cm['fn']} | {cm['tp']} |

- Toplam test örneği: {metrics['total_samples']}
- Pozitif (Anormal): {metrics['positive_samples']}
- Negatif (Normal): {metrics['negative_samples']}
- Eşik değeri: {metrics['threshold']}
"""

    with open(results_dir / "test_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Results saved to {results_dir}")


def evaluate(data_dir: Path, model_path: Path, output_dir: Path):
    """Full evaluation pipeline."""
    logger.info("Loading model...")
    model = tf.keras.models.load_model(model_path)

    logger.info("Loading test data...")
    test_paths, test_labels = load_test_data(data_dir / "splits" / "test.csv", data_dir)
    logger.info(f"Test set: {len(test_paths)} samples")

    logger.info("Running inference...")
    y_prob = predict_test_set(model, test_paths)

    logger.info("Computing metrics...")
    metrics = compute_metrics(test_labels, y_prob)

    # Print results
    logger.info("=" * 50)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 50)
    for k, v in metrics.items():
        if k not in ("confusion_matrix", "total_samples", "positive_samples",
                      "negative_samples", "threshold"):
            logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 50)

    save_results(metrics, output_dir)
    generate_figures(test_labels, y_prob, metrics, output_dir)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate heart sound classifier")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    evaluate(args.data_dir, args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
