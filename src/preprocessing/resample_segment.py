"""
Heart Sound Resampling and Segmentation

Resamples PhysioNet recordings from 2000Hz to 4000Hz using
scipy.signal.resample_poly, then segments all recordings into
5-second (20,000 sample) chunks.

Usage:
    python -m src.preprocessing.resample_segment
    python -m src.preprocessing.resample_segment --dry-run
    python -m src.preprocessing.resample_segment --verify
    python -m src.preprocessing.resample_segment --figures-only
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from fractions import Fraction

from scipy.signal import resample_poly
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_patient_id(row: dict) -> str:
    """Return the real patient ID for patient-based splitting."""
    if row["dataset_source"] == "pascal":
        # PASCAL filenames vary:
        #   normal__103_1305031931979_B.wav
        #   normal_noisynormal_101_1305030823364_B.wav
        # Patient ID is always the first 3-digit number in the stem
        import re
        match = re.search(r'(?<![0-9])(\d{3})(?![0-9])', row["patient_id"])
        if match:
            return match.group(1)
        return row["patient_id"].split("_")[0]
    return row["patient_id"]


def resample_and_segment(config: dict, project_root: Path,
                         dry_run: bool = False) -> Path:
    """Resample PhysioNet to 4000Hz and segment all recordings into 5s chunks."""
    prep = config["preprocessing"]
    target_sr = prep["target_sr"]
    segment_length_s = prep["segment_length_s"]
    segment_samples = target_sr * segment_length_s  # 20000

    metadata_path = project_root / config["paths"]["metadata_csv"]
    segments_dir = project_root / config["paths"]["segments_dir"]
    segments_meta_path = project_root / config["paths"]["segments_metadata_csv"]

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found. Run Stage 1 (download) first.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not dry_run:
        segments_dir.mkdir(parents=True, exist_ok=True)

    segment_rows = []
    stats = {"total": 0, "skipped_short": 0, "skipped_error": 0,
             "segments": 0, "normal": 0, "abnormal": 0}

    for row in tqdm(rows, desc="Processing recordings"):
        stats["total"] += 1
        wav_path = project_root / row["file_path"]
        rec_id = wav_path.stem
        patient_id = extract_patient_id(row)
        binary_label = int(row["binary_label"])
        source = row["dataset_source"]

        if dry_run:
            # Read just metadata to estimate segments
            try:
                info = sf.info(wav_path)
                sr = info.samplerate
                n_frames = info.frames
            except Exception as e:
                logger.warning(f"Cannot read {wav_path}: {e}")
                stats["skipped_error"] += 1
                continue

            if sr != target_sr:
                n_frames = int(n_frames * target_sr / sr)  # post-resample length

            n_seg = n_frames // segment_samples
            # short clips (>=1 s of signal) are zero-padded to a single segment
            if n_seg == 0 and n_frames >= target_sr:
                n_seg = 1
            if n_seg == 0:
                stats["skipped_short"] += 1
            else:
                stats["segments"] += n_seg
                if binary_label == 0:
                    stats["normal"] += n_seg
                else:
                    stats["abnormal"] += n_seg
            continue

        # Read audio
        try:
            audio, sr = sf.read(wav_path, dtype="float64")
        except Exception as e:
            logger.warning(f"Cannot read {wav_path}: {e}")
            stats["skipped_error"] += 1
            continue

        # Mono-mix any multichannel input (new datasets may be stereo)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to target_sr. PhysioNet (2k->4k) and PASCAL (native 4k) keep
        # their exact, thesis-reproducible behavior; new sources use a general
        # rational resampler.
        if source == "physionet":
            if sr != 2000:
                logger.warning(f"Unexpected SR {sr} for PhysioNet file {wav_path}, skipping")
                stats["skipped_error"] += 1
                continue
            audio = resample_poly(audio, up=2, down=1)
        elif source == "pascal":
            if sr != 4000:
                logger.warning(f"Unexpected SR {sr} for PASCAL file {wav_path}, skipping")
                stats["skipped_error"] += 1
                continue
        elif sr != target_sr:
            frac = Fraction(target_sr, sr).limit_denominator(1000)
            audio = resample_poly(audio, up=frac.numerator, down=frac.denominator)

        # Segment. Recordings shorter than one segment but holding >=1 s of
        # signal (e.g. Yaseen ~2-3 s clips) are zero-padded to a single segment
        # rather than dropped.
        if len(audio) < segment_samples:
            if len(audio) < target_sr:
                stats["skipped_short"] += 1
                continue
            audio = np.pad(audio, (0, segment_samples - len(audio)))
        n_segments = len(audio) // segment_samples

        for seg_idx in range(n_segments):
            start = seg_idx * segment_samples
            end = start + segment_samples
            segment = audio[start:end]

            filename = f"{patient_id}_{rec_id}_{seg_idx:03d}.wav"
            seg_path = segments_dir / filename
            sf.write(seg_path, segment, target_sr, subtype="PCM_16")

            rel_path = seg_path.relative_to(project_root).as_posix()
            segment_rows.append({
                "segment_path": rel_path,
                "binary_label": binary_label,
                "patient_id": patient_id,
            })

            stats["segments"] += 1
            if binary_label == 0:
                stats["normal"] += 1
            else:
                stats["abnormal"] += 1

    # Write segments_metadata.csv
    if not dry_run and segment_rows:
        with open(segments_meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["segment_path", "binary_label", "patient_id"])
            writer.writeheader()
            writer.writerows(segment_rows)

    # Report
    logger.info("=" * 50)
    logger.info("SEGMENTATION SUMMARY" + (" (DRY RUN)" if dry_run else ""))
    logger.info("=" * 50)
    logger.info(f"  Recordings processed: {stats['total']}")
    logger.info(f"  Skipped (too short):  {stats['skipped_short']}")
    logger.info(f"  Skipped (error):      {stats['skipped_error']}")
    logger.info(f"  Total segments:       {stats['segments']}")
    logger.info(f"  Normal segments:      {stats['normal']}")
    logger.info(f"  Abnormal segments:    {stats['abnormal']}")
    if not dry_run:
        logger.info(f"  Metadata: {segments_meta_path}")
    logger.info("=" * 50)

    return segments_meta_path


def verify_segments(config: dict, project_root: Path) -> bool:
    """Verify segment files match metadata and have correct properties."""
    segments_dir = project_root / config["paths"]["segments_dir"]
    segments_meta_path = project_root / config["paths"]["segments_metadata_csv"]
    target_sr = config["preprocessing"]["target_sr"]
    segment_samples = target_sr * config["preprocessing"]["segment_length_s"]

    all_ok = True
    logger.info("=" * 50)
    logger.info("SEGMENT VERIFICATION")
    logger.info("=" * 50)

    # Check metadata exists
    if not segments_meta_path.exists():
        logger.error(f"  [FAIL] {segments_meta_path} not found")
        return False

    with open(segments_meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        meta_rows = list(reader)

    meta_count = len(meta_rows)

    # Check file count
    wav_files = list(segments_dir.glob("*.wav"))
    file_count = len(wav_files)
    ok = file_count == meta_count
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_ok = False
    logger.info(f"  [{status}] WAV files: {file_count}, metadata rows: {meta_count}")

    # Spot-check 20 random segments
    import random
    sample_size = min(20, len(wav_files))
    if sample_size > 0:
        samples = random.sample(wav_files, sample_size)
        bad = 0
        for wav in samples:
            try:
                info = sf.info(wav)
                if info.samplerate != target_sr:
                    logger.warning(f"  Wrong SR: {wav} ({info.samplerate})")
                    bad += 1
                elif info.frames != segment_samples:
                    logger.warning(f"  Wrong length: {wav} ({info.frames} samples)")
                    bad += 1
                elif info.channels != 1:
                    logger.warning(f"  Not mono: {wav}")
                    bad += 1
            except Exception as e:
                logger.warning(f"  Cannot read {wav}: {e}")
                bad += 1

        ok = bad == 0
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        logger.info(f"  [{status}] Spot-check: {sample_size - bad}/{sample_size} segments valid "
                     f"(4000Hz, mono, {segment_samples} samples)")

    # Class distribution
    normal = sum(1 for r in meta_rows if r["binary_label"] == "0")
    abnormal = sum(1 for r in meta_rows if r["binary_label"] == "1")
    logger.info(f"  Normal: {normal}  |  Abnormal: {abnormal}  |  Total: {meta_count}")

    # Unique patients
    patients = set(r["patient_id"] for r in meta_rows)
    logger.info(f"  Unique patients: {len(patients)}")

    logger.info("=" * 50)
    overall = "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"
    logger.info(f"  {overall}")
    logger.info("=" * 50)

    return all_ok


def generate_figures(config: dict, project_root: Path):
    """Generate Stage 2 thesis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)
    plt.rcParams["font.family"] = "DejaVu Sans"

    segments_meta_path = project_root / config["paths"]["segments_metadata_csv"]
    segments_dir = project_root / config["paths"]["segments_dir"]
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(segments_meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        meta_rows = list(reader)

    # --- fig_segment_count.png ---
    normal = sum(1 for r in meta_rows if r["binary_label"] == "0")
    abnormal = sum(1 for r in meta_rows if r["binary_label"] == "1")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Normal", "Anormal"], [normal, abnormal],
                  color=sns.color_palette("colorblind")[:2], edgecolor="black", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 50, f"{int(h)}",
                ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_title("Segmentasyon Sonrası Sınıf Başına Segment Sayısı",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Sınıf", fontsize=12)
    ax.set_ylabel("Segment Sayısı", fontsize=12)
    plt.tight_layout()
    path1 = figures_dir / "fig_segment_count.png"
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path1}")

    # --- fig_raw_pcg_example.png ---
    normal_seg = None
    abnormal_seg = None
    for r in meta_rows:
        if r["binary_label"] == "0" and normal_seg is None:
            normal_seg = r
        elif r["binary_label"] == "1" and abnormal_seg is None:
            abnormal_seg = r
        if normal_seg and abnormal_seg:
            break

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    target_sr = config["preprocessing"]["target_sr"]

    for ax, seg_row, title in [
        (axes[0], normal_seg, "Normal"),
        (axes[1], abnormal_seg, "Anormal"),
    ]:
        audio, sr = sf.read(project_root / seg_row["segment_path"], dtype="float64")
        t = np.arange(len(audio)) / sr
        ax.plot(t, audio, linewidth=0.4, color="steelblue")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Zaman (s)", fontsize=11)
        ax.set_xlim(0, 5)

    axes[0].set_ylabel("Genlik", fontsize=11)
    fig.suptitle("5 Saniyelik PCG Segment Örnekleri", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path2 = figures_dir / "fig_raw_pcg_example.png"
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path2}")


def main():
    parser = argparse.ArgumentParser(description="Resample and segment heart sound recordings")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate output without writing files")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing segments")
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
        verify_segments(config, PROJECT_ROOT)
        return

    if args.figures_only:
        generate_figures(config, PROJECT_ROOT)
        return

    resample_and_segment(config, PROJECT_ROOT, dry_run=args.dry_run)
    if not args.dry_run:
        verify_segments(config, PROJECT_ROOT)
        generate_figures(config, PROJECT_ROOT)


if __name__ == "__main__":
    main()
