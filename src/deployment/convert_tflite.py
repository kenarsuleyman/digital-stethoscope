"""
TFLite Conversion and Validation

Converts Keras model to TFLite with float16 quantization.
Runs consistency test comparing Keras vs TFLite predictions on the test set.
Reports model size and inference time.

Usage (on Colab):
    python -m src.deployment.convert_tflite --model-path /content/output/models/best_model.keras --data-dir /content/data --output-dir /content/output
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def convert_to_tflite(model_path: Path, output_path: Path) -> Path:
    """Convert Keras model to TFLite with float16 quantization."""
    logger.info(f"Loading Keras model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    logger.info("Converting to TFLite (float16 quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    logger.info(f"TFLite model saved: {output_path} ({size_mb:.2f} MB)")

    return output_path


def run_tflite_inference(tflite_path: Path, input_data: np.ndarray) -> np.ndarray:
    """Run TFLite inference on a batch of inputs."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    for sample in input_data:
        input_tensor = np.expand_dims(sample, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        predictions.append(output.flatten()[0])

    return np.array(predictions)


def consistency_test(keras_model_path: Path, tflite_path: Path,
                     data_dir: Path, output_dir: Path):
    """Compare Keras and TFLite model predictions on full test set."""
    logger.info("Running consistency test (Keras vs TFLite)...")

    # Load test data
    test_csv = data_dir / "splits" / "test.csv"
    with open(test_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    test_paths = [str(data_dir / r["mel_path"]) for r in rows]
    test_labels = np.array([int(r["binary_label"]) for r in rows])

    # Load all test images
    logger.info(f"Loading {len(test_paths)} test samples...")
    test_images = np.stack([np.load(p).astype(np.float32) for p in test_paths])

    # Keras predictions + timing
    logger.info("Keras inference...")
    keras_model = tf.keras.models.load_model(keras_model_path)
    start = time.time()
    keras_preds = keras_model.predict(test_images, verbose=0).flatten()
    keras_time = (time.time() - start) / len(test_images) * 1000  # ms per sample

    # TFLite predictions + timing
    logger.info("TFLite inference...")
    start = time.time()
    tflite_preds = run_tflite_inference(tflite_path, test_images)
    tflite_time = (time.time() - start) / len(test_images) * 1000  # ms per sample

    # Comparison metrics
    pred_diff = np.abs(keras_preds - tflite_preds)
    keras_acc = np.mean((keras_preds >= 0.5).astype(int) == test_labels)
    tflite_acc = np.mean((tflite_preds >= 0.5).astype(int) == test_labels)
    agreement = np.mean((keras_preds >= 0.5) == (tflite_preds >= 0.5))

    # Model sizes
    keras_size = keras_model_path.stat().st_size / (1024 * 1024)
    tflite_size = tflite_path.stat().st_size / (1024 * 1024)

    results = {
        "keras": {
            "accuracy": float(keras_acc),
            "model_size_mb": float(keras_size),
            "avg_inference_ms": float(keras_time),
        },
        "tflite_fp16": {
            "accuracy": float(tflite_acc),
            "model_size_mb": float(tflite_size),
            "avg_inference_ms": float(tflite_time),
        },
        "comparison": {
            "prediction_agreement": float(agreement),
            "mean_abs_diff": float(np.mean(pred_diff)),
            "max_abs_diff": float(np.max(pred_diff)),
            "size_reduction": f"{(1 - tflite_size / keras_size) * 100:.1f}%",
        },
    }

    # Report
    logger.info("=" * 55)
    logger.info("KERAS vs TFLITE COMPARISON")
    logger.info("=" * 55)
    logger.info(f"  {'Metric':<25} {'Keras':>12} {'TFLite FP16':>12}")
    logger.info(f"  {'-'*50}")
    logger.info(f"  {'Accuracy':<25} {keras_acc:>12.4f} {tflite_acc:>12.4f}")
    logger.info(f"  {'Model size (MB)':<25} {keras_size:>12.2f} {tflite_size:>12.2f}")
    logger.info(f"  {'Avg inference (ms)':<25} {keras_time:>12.2f} {tflite_time:>12.2f}")
    logger.info(f"  {'-'*50}")
    logger.info(f"  Prediction agreement: {agreement:.4f}")
    logger.info(f"  Mean abs diff: {np.mean(pred_diff):.6f}")
    logger.info(f"  Max abs diff:  {np.max(pred_diff):.6f}")
    logger.info(f"  Size reduction: {results['comparison']['size_reduction']}")
    logger.info("=" * 55)

    # Save results
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "tflite_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate comparison figures
    _plot_comparison(results, output_dir / "figures")

    return results


def _plot_comparison(results: dict, figures_dir: Path):
    """Generate comparison bar charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["font.family"] = "DejaVu Sans"
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = sns.color_palette("colorblind")

    # --- fig_keras_vs_tflite.png ---
    metrics = ["Doğruluk"]
    keras_vals = [results["keras"]["accuracy"]]
    tflite_vals = [results["tflite_fp16"]["accuracy"]]

    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6, 5))
    b1 = ax.bar(x - width / 2, keras_vals, width, label="Keras", color=colors[0])
    b2 = ax.bar(x + width / 2, tflite_vals, width, label="TFLite FP16", color=colors[1])

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.4f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Değer", fontsize=12)
    ax.set_title("Keras vs TFLite (FP16) Model Karşılaştırması", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_keras_vs_tflite.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- fig_model_size_inference.png ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Size
    sizes = [results["keras"]["model_size_mb"], results["tflite_fp16"]["model_size_mb"]]
    bars = axes[0].bar(["Keras", "TFLite FP16"], sizes, color=[colors[0], colors[1]],
                       edgecolor="black", linewidth=0.5)
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{bar.get_height():.1f} MB", ha="center", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Model Boyutu (MB)", fontsize=12)
    axes[0].set_title("Model Boyutu", fontsize=13, fontweight="bold")

    # Inference time
    times = [results["keras"]["avg_inference_ms"], results["tflite_fp16"]["avg_inference_ms"]]
    bars = axes[1].bar(["Keras", "TFLite FP16"], times, color=[colors[0], colors[1]],
                       edgecolor="black", linewidth=0.5)
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{bar.get_height():.2f} ms", ha="center", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Ortalama Çıkarım Süresi (ms)", fontsize=12)
    axes[1].set_title("Çıkarım Süresi", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(figures_dir / "fig_model_size_inference.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved: fig_keras_vs_tflite.png, fig_model_size_inference.png")


def main():
    parser = argparse.ArgumentParser(description="Convert to TFLite and validate")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to Keras .keras model")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root data directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--tflite-name", default="heart_sound_classifier_fp16.tflite",
                        help="Output TFLite filename")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    tflite_path = args.output_dir / "models" / args.tflite_name
    convert_to_tflite(args.model_path, tflite_path)
    consistency_test(args.model_path, tflite_path, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
