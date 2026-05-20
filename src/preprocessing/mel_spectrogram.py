"""
Mel Spectrogram Conversion

Converts processed PCG segments to Mel spectrograms:
1. librosa.feature.melspectrogram (sr=4000, n_fft=512, hop=256, n_mels=64, fmin=25, fmax=900)
2. librosa.power_to_db conversion
3. [0, 1] min-max normalization
4. Stack to 3 channels (RGB) and resize to 224x224
5. Save as .npy files

Usage:
    python -m src.preprocessing.mel_spectrogram
    python -m src.preprocessing.mel_spectrogram --dry-run
    python -m src.preprocessing.mel_spectrogram --verify
    python -m src.preprocessing.mel_spectrogram --figures-only
"""

import argparse
import csv
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def audio_to_mel_image(audio: np.ndarray, sr: int, mel_config: dict,
                       image_size: tuple) -> np.ndarray:
    """
    Convert audio to a 224x224x3 Mel spectrogram image (float32, [0,1]).

    Returns: np.ndarray of shape (H, W, 3), dtype float32
    """
    # Mel spectrogram
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=mel_config["n_fft"],
        hop_length=mel_config["hop_length"],
        n_mels=mel_config["n_mels"],
        fmin=mel_config["fmin"],
        fmax=mel_config["fmax"],
    )

    # Convert to dB
    S_db = librosa.power_to_db(S, ref=np.max)

    # Min-max normalize to [0, 1]
    s_min = S_db.min()
    s_max = S_db.max()
    if s_max - s_min > 0:
        S_norm = (S_db - s_min) / (s_max - s_min)
    else:
        S_norm = np.zeros_like(S_db)

    S_norm = S_norm.astype(np.float32)

    # Resize to target image size using PIL
    from PIL import Image
    img = Image.fromarray(S_norm)
    img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    S_resized = np.array(img, dtype=np.float32)

    # Stack to 3 channels (RGB)
    mel_rgb = np.stack([S_resized, S_resized, S_resized], axis=-1)

    return mel_rgb


def convert_mel_spectrograms(config: dict, project_root: Path,
                             dry_run: bool = False) -> Path:
    """Convert all processed segments to Mel spectrogram .npy files."""
    mel_config = config["mel_spectrogram"]
    target_sr = config["preprocessing"]["target_sr"]
    image_size = tuple(mel_config["image_size"])

    processed_meta_path = project_root / config["paths"]["processed_metadata_csv"]
    mel_dir = project_root / config["paths"]["mel_dir"]
    mel_meta_path = project_root / config["paths"]["mel_metadata_csv"]

    if not processed_meta_path.exists():
        raise FileNotFoundError(
            f"{processed_meta_path} not found. Run Stage 3 (clean_filter) first.")

    with open(processed_meta_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if dry_run:
        # Estimate: each .npy is 224*224*3*4 bytes = ~588 KB
        n = len(rows)
        size_mb = n * 224 * 224 * 3 * 4 / (1024 * 1024)
        logger.info(f"[DRY RUN] Would process {n} segments")
        logger.info(f"[DRY RUN] Output shape per file: ({image_size[0]}, {image_size[1]}, 3)")
        logger.info(f"[DRY RUN] Estimated total size: {size_mb:.0f} MB")
        return mel_meta_path

    mel_dir.mkdir(parents=True, exist_ok=True)
    mel_rows = []
    errors = 0

    for row in tqdm(rows, desc="Converting to Mel"):
        wav_path = project_root / row["segment_path"]
        npy_name = Path(row["segment_path"]).stem + ".npy"
        npy_path = mel_dir / npy_name

        try:
            audio, sr = sf.read(wav_path, dtype="float64")
            mel_rgb = audio_to_mel_image(audio, sr, mel_config, image_size)
            np.save(npy_path, mel_rgb)

            rel_path = npy_path.relative_to(project_root).as_posix()
            mel_rows.append({
                "mel_path": rel_path,
                "binary_label": row["binary_label"],
                "patient_id": row["patient_id"],
            })
        except Exception as e:
            logger.warning(f"Error processing {wav_path}: {e}")
            errors += 1

    # Write metadata
    with open(mel_meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mel_path", "binary_label", "patient_id"])
        writer.writeheader()
        writer.writerows(mel_rows)

    normal = sum(1 for r in mel_rows if r["binary_label"] == "0")
    abnormal = sum(1 for r in mel_rows if r["binary_label"] == "1")

    logger.info("=" * 50)
    logger.info("MEL SPECTROGRAM SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Input segments:  {len(rows)}")
    logger.info(f"  Converted:       {len(mel_rows)}")
    logger.info(f"  Errors:          {errors}")
    logger.info(f"  Normal:          {normal}")
    logger.info(f"  Abnormal:        {abnormal}")
    logger.info(f"  Shape per file:  ({image_size[0]}, {image_size[1]}, 3)")
    logger.info(f"  Metadata:        {mel_meta_path}")
    logger.info("=" * 50)

    return mel_meta_path


def verify_mel(config: dict, project_root: Path) -> bool:
    """Verify Mel spectrogram .npy files."""
    mel_dir = project_root / config["paths"]["mel_dir"]
    mel_meta_path = project_root / config["paths"]["mel_metadata_csv"]
    image_size = tuple(config["mel_spectrogram"]["image_size"])

    all_ok = True
    logger.info("=" * 50)
    logger.info("MEL SPECTROGRAM VERIFICATION")
    logger.info("=" * 50)

    if not mel_meta_path.exists():
        logger.error(f"  [FAIL] {mel_meta_path} not found")
        return False

    with open(mel_meta_path, "r", encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))

    npy_files = list(mel_dir.glob("*.npy"))
    ok = len(npy_files) == len(meta_rows)
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_ok = False
    logger.info(f"  [{status}] .npy files: {len(npy_files)}, metadata rows: {len(meta_rows)}")

    # Spot-check
    import random
    sample_size = min(20, len(npy_files))
    if sample_size > 0:
        samples = random.sample(npy_files, sample_size)
        bad = 0
        for npy in samples:
            try:
                data = np.load(npy)
                if data.shape != (image_size[0], image_size[1], 3):
                    bad += 1
                elif data.dtype != np.float32:
                    bad += 1
                elif data.min() < -0.01 or data.max() > 1.01:
                    bad += 1
            except Exception:
                bad += 1

        ok = bad == 0
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        logger.info(f"  [{status}] Spot-check: {sample_size - bad}/{sample_size} valid "
                     f"(shape={image_size}, float32, [0,1])")

    normal = sum(1 for r in meta_rows if r["binary_label"] == "0")
    abnormal = sum(1 for r in meta_rows if r["binary_label"] == "1")
    logger.info(f"  Normal: {normal}  |  Abnormal: {abnormal}  |  Total: {len(meta_rows)}")

    logger.info("=" * 50)
    overall = "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"
    logger.info(f"  {overall}")
    logger.info("=" * 50)
    return all_ok


def generate_figures(config: dict, project_root: Path):
    """Generate Stage 4 thesis figures: Mel spectrogram examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", palette="colorblind", font_scale=1.1)
    plt.rcParams["font.family"] = "DejaVu Sans"

    mel_config = config["mel_spectrogram"]
    target_sr = config["preprocessing"]["target_sr"]
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    processed_meta_path = project_root / config["paths"]["processed_metadata_csv"]
    with open(processed_meta_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Find one normal, one abnormal
    normal_row = next(r for r in rows if r["binary_label"] == "0")
    abnormal_row = next(r for r in rows if r["binary_label"] == "1")

    def compute_mel_db(row):
        audio, sr = sf.read(project_root / row["segment_path"], dtype="float64")
        S = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_fft=mel_config["n_fft"],
            hop_length=mel_config["hop_length"],
            n_mels=mel_config["n_mels"],
            fmin=mel_config["fmin"],
            fmax=mel_config["fmax"],
        )
        return librosa.power_to_db(S, ref=np.max)

    mel_normal = compute_mel_db(normal_row)
    mel_abnormal = compute_mel_db(abnormal_row)

    hop = mel_config["hop_length"]

    # --- fig_mel_normal.png ---
    fig, ax = plt.subplots(figsize=(8, 5))
    img = librosa.display.specshow(
        mel_normal, sr=target_sr, hop_length=hop,
        x_axis="time", y_axis="mel", fmin=mel_config["fmin"],
        fmax=mel_config["fmax"], ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Güç (dB)")
    ax.set_title("Normal — Mel Spektrogram", fontsize=13, fontweight="bold")
    ax.set_xlabel("Zaman (s)", fontsize=11)
    ax.set_ylabel("Frekans (Hz)", fontsize=11)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_mel_normal.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: fig_mel_normal.png")

    # --- fig_mel_abnormal.png ---
    fig, ax = plt.subplots(figsize=(8, 5))
    img = librosa.display.specshow(
        mel_abnormal, sr=target_sr, hop_length=hop,
        x_axis="time", y_axis="mel", fmin=mel_config["fmin"],
        fmax=mel_config["fmax"], ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Güç (dB)")
    ax.set_title("Anormal — Mel Spektrogram", fontsize=13, fontweight="bold")
    ax.set_xlabel("Zaman (s)", fontsize=11)
    ax.set_ylabel("Frekans (Hz)", fontsize=11)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_mel_abnormal.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: fig_mel_abnormal.png")

    # --- fig_mel_comparison.png ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    img0 = librosa.display.specshow(
        mel_normal, sr=target_sr, hop_length=hop,
        x_axis="time", y_axis="mel", fmin=mel_config["fmin"],
        fmax=mel_config["fmax"], ax=axes[0])
    axes[0].set_title("Normal", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Zaman (s)", fontsize=11)
    axes[0].set_ylabel("Frekans (Hz)", fontsize=11)
    fig.colorbar(img0, ax=axes[0], format="%+2.0f dB")

    img1 = librosa.display.specshow(
        mel_abnormal, sr=target_sr, hop_length=hop,
        x_axis="time", y_axis="mel", fmin=mel_config["fmin"],
        fmax=mel_config["fmax"], ax=axes[1])
    axes[1].set_title("Anormal", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Zaman (s)", fontsize=11)
    axes[1].set_ylabel("")
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")

    fig.suptitle("Normal ve Anormal Mel Spektrogram Karşılaştırması",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_mel_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: fig_mel_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Convert segments to Mel spectrograms")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate output without converting")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing Mel spectrograms")
    parser.add_argument("--figures-only", action="store_true",
                        help="Only generate thesis figures")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)

    if args.verify:
        verify_mel(config, PROJECT_ROOT)
        return

    if args.figures_only:
        generate_figures(config, PROJECT_ROOT)
        return

    convert_mel_spectrograms(config, PROJECT_ROOT, dry_run=args.dry_run)
    if not args.dry_run:
        verify_mel(config, PROJECT_ROOT)
        generate_figures(config, PROJECT_ROOT)


if __name__ == "__main__":
    main()
