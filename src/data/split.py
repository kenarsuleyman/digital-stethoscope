"""
Patient-Based Data Splitting

Splits Mel spectrogram data into train/val/test sets (70/15/15)
using GroupShuffleSplit on patient_id to prevent data leakage.
Computes balanced class weights for training.

Usage:
    python -m src.data.split
    python -m src.data.split --verify
    python -m src.data.split --figures-only
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_data(config: dict, project_root: Path) -> dict:
    """
    Split mel metadata into train/val/test by patient_id.

    Strategy: Two-stage GroupShuffleSplit
    1. Split into train (70%) and temp (30%) by patient
    2. Split temp into val (50% of temp = 15%) and test (50% of temp = 15%) by patient

    Returns dict with split info.
    """
    mel_meta_path = project_root / config["paths"]["mel_metadata_csv"]
    splits_dir = project_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not mel_meta_path.exists():
        raise FileNotFoundError(
            f"{mel_meta_path} not found. Run Stage 4 (mel_spectrogram) first.")

    with open(mel_meta_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    mel_paths = np.array([r["mel_path"] for r in rows])
    labels = np.array([int(r["binary_label"]) for r in rows])
    patient_ids = np.array([r["patient_id"] for r in rows])

    # Stage 1: train (70%) vs temp (30%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(mel_paths, labels, groups=patient_ids))

    # Stage 2: split temp into val (50%) and test (50%) → 15% each of total
    temp_paths = mel_paths[temp_idx]
    temp_labels = labels[temp_idx]
    temp_patients = patient_ids[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_rel_idx, test_rel_idx = next(gss2.split(temp_paths, temp_labels, groups=temp_patients))
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    # Write split CSVs
    fieldnames = ["mel_path", "binary_label", "patient_id"]
    split_info = {}

    for name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_rows = [rows[i] for i in indices]
        split_path = splits_dir / f"{name}.csv"

        with open(split_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)

        n_normal = sum(1 for r in split_rows if r["binary_label"] == "0")
        n_abnormal = sum(1 for r in split_rows if r["binary_label"] == "1")
        n_patients = len(set(r["patient_id"] for r in split_rows))

        split_info[name] = {
            "count": len(split_rows),
            "normal": n_normal,
            "abnormal": n_abnormal,
            "patients": n_patients,
            "path": str(split_path),
        }

    # Compute class weights from training set
    train_labels = labels[train_idx]
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=train_labels)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    weights_path = project_root / "configs" / "class_weights.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(class_weights, f, indent=2)

    # Verify no patient leakage
    train_patients = set(patient_ids[train_idx])
    val_patients = set(patient_ids[val_idx])
    test_patients = set(patient_ids[test_idx])

    leak_tv = train_patients & val_patients
    leak_tt = train_patients & test_patients
    leak_vt = val_patients & test_patients

    # Report
    logger.info("=" * 55)
    logger.info("PATIENT-BASED SPLIT SUMMARY")
    logger.info("=" * 55)
    logger.info(f"  {'Set':<8} {'Segments':>10} {'Normal':>8} {'Abnormal':>10} {'Patients':>10}")
    logger.info(f"  {'-'*48}")
    for name in ["train", "val", "test"]:
        s = split_info[name]
        pct = s['count'] / len(rows) * 100
        logger.info(f"  {name:<8} {s['count']:>10} {s['normal']:>8} {s['abnormal']:>10} "
                     f"{s['patients']:>10}  ({pct:.1f}%)")
    logger.info(f"  {'-'*48}")
    logger.info(f"  {'Total':<8} {len(rows):>10}")
    logger.info("")
    logger.info(f"  Class weights: {class_weights}")
    logger.info(f"  Saved to: {weights_path}")
    logger.info("")
    logger.info(f"  Patient leakage train↔val:  {len(leak_tv)} patients")
    logger.info(f"  Patient leakage train↔test: {len(leak_tt)} patients")
    logger.info(f"  Patient leakage val↔test:   {len(leak_vt)} patients")
    if leak_tv or leak_tt or leak_vt:
        logger.warning("  ⚠ PATIENT LEAKAGE DETECTED!")
    else:
        logger.info("  ✓ No patient leakage — all splits are patient-disjoint")
    logger.info("=" * 55)

    return split_info


def verify_split(config: dict, project_root: Path) -> bool:
    """Verify split files and check for patient leakage."""
    splits_dir = project_root / "splits"
    all_ok = True

    logger.info("=" * 55)
    logger.info("SPLIT VERIFICATION")
    logger.info("=" * 55)

    all_patients = {"train": set(), "val": set(), "test": set()}
    total = 0

    for name in ["train", "val", "test"]:
        split_path = splits_dir / f"{name}.csv"
        if not split_path.exists():
            logger.error(f"  [FAIL] {split_path} not found")
            all_ok = False
            continue

        with open(split_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        patients = set(r["patient_id"] for r in rows)
        all_patients[name] = patients
        total += len(rows)

        # Verify mel files exist
        missing = 0
        for r in rows[:50]:  # spot-check first 50
            if not (project_root / r["mel_path"]).exists():
                missing += 1

        status = "PASS" if missing == 0 else "FAIL"
        if missing > 0:
            all_ok = False
        logger.info(f"  [{status}] {name}: {len(rows)} rows, {len(patients)} patients, "
                     f"{missing} missing files (checked 50)")

    # Patient leakage
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = all_patients[a] & all_patients[b]
        ok = len(overlap) == 0
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        logger.info(f"  [{status}] {a}↔{b} patient overlap: {len(overlap)}")

    # Class weights
    weights_path = project_root / "configs" / "class_weights.json"
    ok = weights_path.exists()
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_ok = False
    else:
        with open(weights_path) as f:
            w = json.load(f)
        logger.info(f"  [{status}] class_weights.json: {w}")

    logger.info("=" * 55)
    overall = "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"
    logger.info(f"  {overall}")
    logger.info("=" * 55)
    return all_ok


def generate_figures(config: dict, project_root: Path):
    """Generate fig_split_distribution.png — stacked bar chart of splits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)
    plt.rcParams["font.family"] = "DejaVu Sans"

    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = project_root / "splits"

    split_data = {}
    for name in ["train", "val", "test"]:
        with open(splits_dir / f"{name}.csv", "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        split_data[name] = {
            "Normal": sum(1 for r in rows if r["binary_label"] == "0"),
            "Anormal": sum(1 for r in rows if r["binary_label"] == "1"),
        }

    labels = ["Eğitim", "Doğrulama", "Test"]
    keys = ["train", "val", "test"]
    normal_counts = [split_data[k]["Normal"] for k in keys]
    abnormal_counts = [split_data[k]["Anormal"] for k in keys]

    x = np.arange(len(labels))
    width = 0.5

    colors = sns.color_palette("colorblind")

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x, normal_counts, width, label="Normal", color=colors[0], edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, abnormal_counts, width, bottom=normal_counts, label="Anormal", color=colors[1], edgecolor="black", linewidth=0.5)

    # Add count labels
    for bar, count in zip(bars1, normal_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    f"{count}", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    for bar, count, bottom in zip(bars2, abnormal_counts, normal_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bottom + bar.get_height() / 2,
                    f"{count}", ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    # Add total on top
    for i, (n, a) in enumerate(zip(normal_counts, abnormal_counts)):
        total = n + a
        ax.text(i, total + 100, f"Toplam: {total}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_title("Eğitim / Doğrulama / Test Setlerinde Sınıf Dağılımı",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Veri Seti", fontsize=12)
    ax.set_ylabel("Segment Sayısı", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Sınıf", fontsize=11, title_fontsize=12)

    plt.tight_layout()
    path = figures_dir / "fig_split_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Patient-based train/val/test split")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing splits")
    parser.add_argument("--figures-only", action="store_true",
                        help="Only generate thesis figure")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)

    if args.verify:
        verify_split(config, PROJECT_ROOT)
        return

    if args.figures_only:
        generate_figures(config, PROJECT_ROOT)
        return

    split_data(config, PROJECT_ROOT)
    verify_split(config, PROJECT_ROOT)
    generate_figures(config, PROJECT_ROOT)


if __name__ == "__main__":
    main()
