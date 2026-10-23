"""
Two-Stage Training Pipeline

Stage 1: Base frozen, Adam lr=1e-3, 10 epochs
Stage 2: Last 4 blocks unfrozen, Adam lr=1e-5, 30 epochs

Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
Class weights applied via model.fit()

Usage (on Colab):
    python -m src.training.train --data-dir /content/data --output-dir /content/output
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

logger = logging.getLogger(__name__)


def load_split_data(split_csv: Path, project_root: Path):
    """Load mel .npy paths and labels from a split CSV."""
    with open(split_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    paths = [str(project_root / r["mel_path"]) for r in rows]
    labels = [int(r["binary_label"]) for r in rows]
    return paths, labels


def make_augmenter(aug_cfg: dict, n_frames: int = 79, n_mels: int = 64, size: int = 224):
    """On-the-fly augmentation of the precomputed 224x224 Mel images (thesis Section 3.1.4).

    The methods of src/preprocessing/augmentation.py are applied in the image domain, scaled from Mel
    frames / bands to pixels (5 s at 4 kHz with hop 256 = 79 frames over 224 px; 64 bands over 224 px):
      - time shift with probability p, up to +/- time_shift_ms (zero fill at the edge)
      - then one of: time mask (frames), frequency mask (bands), amplitude scaling, chosen at random
    """
    px_f = size / n_frames
    px_b = size / n_mels
    shift_px = int(round(aug_cfg["time_shift_ms"] / 1000 * 4000 / 256 * px_f))
    tm = [int(round(v * px_f)) for v in aug_cfg["time_mask_frames"]]
    fm = [int(round(v * px_b)) for v in aug_cfg["freq_mask_bands"]]
    lo, hi = aug_cfg["amp_scale"]
    p_shift = aug_cfg["time_shift_prob"]
    idx = tf.range(size)

    def time_shift(img):
        s = tf.random.uniform((), -shift_px, shift_px + 1, dtype=tf.int32)
        keep = tf.cast((idx >= s) & (idx < size + s), img.dtype)
        return tf.roll(img, s, axis=1) * keep[None, :, None]

    def band_mask(img, lmin, lmax, axis):
        n = tf.random.uniform((), lmin, lmax + 1, dtype=tf.int32)
        start = tf.random.uniform((), 0, size - n + 1, dtype=tf.int32)
        m = 1.0 - tf.cast((idx >= start) & (idx < start + n), img.dtype)
        return img * (m[None, :, None] if axis == 1 else m[:, None, None])

    def augment(img, label):
        img = tf.cond(tf.random.uniform(()) < p_shift, lambda: time_shift(img), lambda: img)
        choice = tf.random.uniform((), 0, 3, dtype=tf.int32)
        img = tf.switch_case(choice, [lambda: band_mask(img, tm[0], tm[1], axis=1),
                                      lambda: band_mask(img, fm[0], fm[1], axis=0),
                                      lambda: img * tf.random.uniform((), lo, hi)])
        return tf.clip_by_value(img, 0.0, 1.0), label

    return augment


def create_dataset(paths: list, labels: list, batch_size: int,
                   shuffle: bool = False, augment: bool = False, aug_cfg: dict = None) -> tf.data.Dataset:
    """Create a tf.data.Dataset from .npy file paths and labels (augment: training set only)."""

    def load_npy(path_tensor, label):
        def _load(p):
            arr = np.load(p.numpy().decode("utf-8"))
            return arr.astype(np.float32)

        image = tf.py_function(_load, [path_tensor], tf.float32)
        image.set_shape([224, 224, 3])
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))

    ds = ds.map(load_npy, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(make_augmenter(aug_cfg), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def train_model(data_dir: Path, output_dir: Path, config: dict):
    """Run two-stage training."""
    from src.models.build_model import build_stage1_model, unfreeze_for_stage2

    splits_dir = data_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]

    # Load class weights
    weights_path = data_dir / "configs" / "class_weights.json"
    with open(weights_path) as f:
        class_weights = {int(k): v for k, v in json.load(f).items()}
    logger.info(f"Class weights: {class_weights}")

    # Load data
    logger.info("Loading datasets...")
    train_paths, train_labels = load_split_data(splits_dir / "train.csv", data_dir)
    val_paths, val_labels = load_split_data(splits_dir / "val.csv", data_dir)

    aug_cfg = train_cfg.get("augmentation", {})
    train_ds = create_dataset(train_paths, train_labels, batch_size, shuffle=True,
                              augment=aug_cfg.get("enabled", False), aug_cfg=aug_cfg)
    val_ds = create_dataset(val_paths, val_labels, batch_size)
    logger.info(f"Augmentation: {aug_cfg if aug_cfg.get('enabled') else 'off'}")

    logger.info(f"Train: {len(train_paths)} samples, Val: {len(val_paths)} samples")

    # ===================== STAGE 1 =====================
    logger.info("=" * 50)
    logger.info("STAGE 1: Feature extraction (base frozen)")
    logger.info("=" * 50)

    model = build_stage1_model()
    model.compile(
        optimizer=Adam(learning_rate=train_cfg["stage1"]["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()

    stage1_callbacks = [
        ModelCheckpoint(
            str(models_dir / "best_stage1.keras"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        CSVLogger(str(logs_dir / "stage1_log.csv")),
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_cfg["stage1"]["epochs"],
        class_weight=class_weights,
        callbacks=stage1_callbacks,
        verbose=1,
    )

    # ===================== STAGE 2 =====================
    logger.info("=" * 50)
    logger.info("STAGE 2: Fine-tuning (last 4 blocks unfrozen)")
    logger.info("=" * 50)

    # Load best stage 1 weights
    model.load_weights(str(models_dir / "best_stage1.keras"))
    model = unfreeze_for_stage2(model, n_blocks=train_cfg["stage2"]["unfreeze_blocks"])

    model.compile(
        optimizer=Adam(learning_rate=train_cfg["stage2"]["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    stage2_callbacks = [
        ModelCheckpoint(
            str(models_dir / "best_model.keras"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=train_cfg["reduce_lr_factor"],
            patience=train_cfg["reduce_lr_patience"],
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(str(logs_dir / "stage2_log.csv")),
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_cfg["stage2"]["epochs"],
        class_weight=class_weights,
        callbacks=stage2_callbacks,
        verbose=1,
    )

    # Save final model
    model.save(str(models_dir / "final_model.keras"))
    logger.info(f"Final model saved to: {models_dir / 'final_model.keras'}")

    # Generate training curves
    _plot_training_curves(history1, history2, figures_dir)

    return model, history1, history2


def _plot_training_curves(history1, history2, figures_dir: Path):
    """Generate fig_training_curves.png with stage transition marker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"

    # Combine histories
    s1_epochs = len(history1.history["loss"])

    all_loss = history1.history["loss"] + history2.history["loss"]
    all_val_loss = history1.history["val_loss"] + history2.history["val_loss"]
    all_acc = history1.history["accuracy"] + history2.history["accuracy"]
    all_val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    epochs = range(1, len(all_loss) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Loss
    axes[0].plot(epochs, all_loss, "b-", label="Eğitim Kaybı", linewidth=1.5)
    axes[0].plot(epochs, all_val_loss, "r-", label="Doğrulama Kaybı", linewidth=1.5)
    axes[0].axvline(x=s1_epochs + 0.5, color="gray", linestyle="--", alpha=0.7)
    axes[0].text(s1_epochs + 0.5, max(all_loss) * 0.9, "  Fine-tuning\n  başlangıcı",
                 fontsize=9, color="gray")
    axes[0].set_ylabel("Kayıp (Loss)", fontsize=12)
    axes[0].set_title("Eğitim ve Doğrulama Kayıp/Doğruluk Eğrileri",
                      fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, all_acc, "b-", label="Eğitim Doğruluğu", linewidth=1.5)
    axes[1].plot(epochs, all_val_acc, "r-", label="Doğrulama Doğruluğu", linewidth=1.5)
    axes[1].axvline(x=s1_epochs + 0.5, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Doğruluk (Accuracy)", fontsize=12)
    axes[1].set_xlabel("Dönem (Epoch)", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = figures_dir / "fig_training_curves.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # Also save LR schedule if available
    if "lr" in history2.history:
        fig, ax = plt.subplots(figsize=(10, 4))
        lr_all = history1.history.get("lr", []) + history2.history["lr"]
        ax.plot(range(1, len(lr_all) + 1), lr_all, "g-", linewidth=1.5)
        ax.axvline(x=s1_epochs + 0.5, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Dönem (Epoch)", fontsize=12)
        ax.set_ylabel("Öğrenme Oranı", fontsize=12)
        ax.set_title("Öğrenme Oranı Değişimi", fontsize=13, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "fig_lr_schedule.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {figures_dir / 'fig_lr_schedule.png'}")


def main():
    parser = argparse.ArgumentParser(description="Train heart sound classifier")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root directory containing data/, splits/, configs/")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for models, logs, figures")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.yaml (default: data-dir/configs/config.yaml)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    import yaml
    config_path = args.config or (args.data_dir / "configs" / "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_model(args.data_dir, args.output_dir, config)


if __name__ == "__main__":
    main()
