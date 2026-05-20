"""
Heart Sound Cleaning and Filtering

1. RMS energy thresholding — reject segments with RMS < 0.1*median or > 10*median
2. Butterworth bandpass filter — 4th order, 25-900 Hz, zero-phase (sosfiltfilt)
3. Amplitude normalization — scale each segment to [-1, +1]

Usage:
    python -m src.preprocessing.clean_filter
    python -m src.preprocessing.clean_filter --dry-run
    python -m src.preprocessing.clean_filter --verify
    python -m src.preprocessing.clean_filter --figures-only
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def design_bandpass(low: float, high: float, sr: int, order: int):
    """Design a Butterworth bandpass filter, return SOS coefficients."""
    nyq = sr / 2.0
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def clean_and_filter(config: dict, project_root: Path,
                     dry_run: bool = False) -> Path:
    """Apply RMS thresholding, bandpass filter, and normalization."""
    prep = config["preprocessing"]
    target_sr = prep["target_sr"]
    bp_low = prep["bandpass_low"]
    bp_high = prep["bandpass_high"]
    bp_order = prep["butterworth_order"]
    rms_low_factor = prep["rms_low_factor"]
    rms_high_factor = prep["rms_high_factor"]

    segments_meta_path = project_root / config["paths"]["segments_metadata_csv"]
    processed_dir = project_root / config["paths"]["processed_dir"]
    processed_meta_path = project_root / config["paths"]["processed_metadata_csv"]
    rejected_log_path = project_root / config["paths"]["rejected_log"]

    if not segments_meta_path.exists():
        raise FileNotFoundError(
            f"{segments_meta_path} not found. Run Stage 2 (resample_segment) first.")

    with open(segments_meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # --- Pass 1: Compute RMS for all segments ---
    logger.info("Pass 1: Computing RMS for all segments...")
    rms_values = []
    for row in tqdm(rows, desc="Computing RMS"):
        wav_path = project_root / row["segment_path"]
        try:
            audio, sr = sf.read(wav_path, dtype="float64")
            rms_values.append(compute_rms(audio))
        except Exception as e:
            logger.warning(f"Cannot read {wav_path}: {e}")
            rms_values.append(0.0)

    rms_array = np.array(rms_values)
    median_rms = float(np.median(rms_array[rms_array > 0]))
    low_thresh = rms_low_factor * median_rms
    high_thresh = rms_high_factor * median_rms

    logger.info(f"  Median RMS: {median_rms:.6f}")
    logger.info(f"  Low threshold (0.1x): {low_thresh:.6f}")
    logger.info(f"  High threshold (10x): {high_thresh:.6f}")

    # Count rejections
    rejected_low = int(np.sum(rms_array < low_thresh))
    rejected_high = int(np.sum(rms_array > high_thresh))
    logger.info(f"  Rejected (too quiet): {rejected_low}")
    logger.info(f"  Rejected (too loud):  {rejected_high}")
    logger.info(f"  Kept: {len(rows) - rejected_low - rejected_high}/{len(rows)}")

    if dry_run:
        logger.info("[DRY RUN] No files written.")
        return processed_meta_path

    # --- Pass 2: Filter, normalize, and save ---
    logger.info("Pass 2: Filtering, normalizing, and saving...")
    processed_dir.mkdir(parents=True, exist_ok=True)

    sos = design_bandpass(bp_low, bp_high, target_sr, bp_order)

    processed_rows = []
    rejected_entries = []

    for row, rms in tqdm(zip(rows, rms_values), total=len(rows), desc="Processing"):
        wav_path = project_root / row["segment_path"]
        filename = Path(row["segment_path"]).name

        # RMS thresholding
        if rms < low_thresh:
            rejected_entries.append(f"{filename}\tREJECTED_LOW_RMS\t{rms:.6f}")
            continue
        if rms > high_thresh:
            rejected_entries.append(f"{filename}\tREJECTED_HIGH_RMS\t{rms:.6f}")
            continue

        try:
            audio, sr = sf.read(wav_path, dtype="float64")
        except Exception as e:
            rejected_entries.append(f"{filename}\tREAD_ERROR\t{e}")
            continue

        # Bandpass filter
        audio = sosfiltfilt(sos, audio)

        # Normalize to [-1, +1]
        max_abs = np.max(np.abs(audio))
        if max_abs > 0:
            audio = audio / max_abs

        # Save
        out_path = processed_dir / filename
        sf.write(out_path, audio, target_sr, subtype="PCM_16")

        rel_path = out_path.relative_to(project_root).as_posix()
        processed_rows.append({
            "segment_path": rel_path,
            "binary_label": row["binary_label"],
            "patient_id": row["patient_id"],
        })

    # Write processed metadata
    with open(processed_meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["segment_path", "binary_label", "patient_id"])
        writer.writeheader()
        writer.writerows(processed_rows)

    # Write rejected log
    with open(rejected_log_path, "w", encoding="utf-8") as f:
        f.write("filename\treason\tvalue\n")
        for entry in rejected_entries:
            f.write(entry + "\n")

    # Summary
    kept = len(processed_rows)
    rejected = len(rejected_entries)
    normal = sum(1 for r in processed_rows if r["binary_label"] == "0")
    abnormal = sum(1 for r in processed_rows if r["binary_label"] == "1")

    logger.info("=" * 50)
    logger.info("CLEANING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Input segments:    {len(rows)}")
    logger.info(f"  Rejected:          {rejected}")
    logger.info(f"  Kept (processed):  {kept}")
    logger.info(f"  Normal:            {normal}")
    logger.info(f"  Abnormal:          {abnormal}")
    logger.info(f"  Rejected log:      {rejected_log_path}")
    logger.info(f"  Processed meta:    {processed_meta_path}")
    logger.info("=" * 50)

    return processed_meta_path


def verify_processed(config: dict, project_root: Path) -> bool:
    """Verify processed segments."""
    processed_dir = project_root / config["paths"]["processed_dir"]
    processed_meta_path = project_root / config["paths"]["processed_metadata_csv"]
    target_sr = config["preprocessing"]["target_sr"]
    segment_samples = target_sr * config["preprocessing"]["segment_length_s"]

    all_ok = True
    logger.info("=" * 50)
    logger.info("PROCESSED SEGMENT VERIFICATION")
    logger.info("=" * 50)

    if not processed_meta_path.exists():
        logger.error(f"  [FAIL] {processed_meta_path} not found")
        return False

    with open(processed_meta_path, "r", encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))

    wav_files = list(processed_dir.glob("*.wav"))
    ok = len(wav_files) == len(meta_rows)
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_ok = False
    logger.info(f"  [{status}] WAV files: {len(wav_files)}, metadata rows: {len(meta_rows)}")

    # Spot-check
    import random
    sample_size = min(20, len(wav_files))
    if sample_size > 0:
        samples = random.sample(wav_files, sample_size)
        bad = 0
        for wav in samples:
            try:
                audio, sr = sf.read(wav, dtype="float64")
                if sr != target_sr:
                    bad += 1
                elif len(audio) != segment_samples:
                    bad += 1
                elif np.max(np.abs(audio)) > 1.001:  # small tolerance for PCM_16 rounding
                    bad += 1
            except Exception:
                bad += 1

        ok = bad == 0
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        logger.info(f"  [{status}] Spot-check: {sample_size - bad}/{sample_size} valid "
                     f"(4000Hz, 20000 samples, normalized)")

    normal = sum(1 for r in meta_rows if r["binary_label"] == "0")
    abnormal = sum(1 for r in meta_rows if r["binary_label"] == "1")
    logger.info(f"  Normal: {normal}  |  Abnormal: {abnormal}  |  Total: {len(meta_rows)}")

    logger.info("=" * 50)
    overall = "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"
    logger.info(f"  {overall}")
    logger.info("=" * 50)
    return all_ok


def generate_figures(config: dict, project_root: Path):
    """Generate Stage 3 thesis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)
    plt.rcParams["font.family"] = "DejaVu Sans"

    prep = config["preprocessing"]
    target_sr = prep["target_sr"]
    bp_low = prep["bandpass_low"]
    bp_high = prep["bandpass_high"]
    bp_order = prep["butterworth_order"]
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    sos = design_bandpass(bp_low, bp_high, target_sr, bp_order)

    # --- fig_filter_before_after.png ---
    # Pick a segment from data/segments/ (raw, before filtering)
    segments_meta_path = project_root / config["paths"]["segments_metadata_csv"]
    with open(segments_meta_path, "r", encoding="utf-8") as f:
        seg_rows = list(csv.DictReader(f))

    # Find an abnormal segment for a more interesting waveform
    example_row = next((r for r in seg_rows if r["binary_label"] == "1"), seg_rows[0])
    raw_audio, sr = sf.read(project_root / example_row["segment_path"], dtype="float64")
    filtered_audio = sosfiltfilt(sos, raw_audio)

    t = np.arange(len(raw_audio)) / target_sr

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t, raw_audio, linewidth=0.4, color="steelblue")
    axes[0].set_title("Ham PCG Segmenti (Filtreleme Öncesi)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Genlik", fontsize=11)

    axes[1].plot(t, filtered_audio, linewidth=0.4, color="darkorange")
    axes[1].set_title(f"Butterworth Bant Geçiren Filtre Sonrası ({bp_low}–{bp_high} Hz)",
                      fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Genlik", fontsize=11)
    axes[1].set_xlabel("Zaman (s)", fontsize=11)

    plt.tight_layout()
    path1 = figures_dir / "fig_filter_before_after.png"
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path1}")

    # --- fig_frequency_response.png ---
    from scipy.signal import sosfreqz

    w, h = sosfreqz(sos, worN=2048, fs=target_sr)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Magnitude
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    axes[0].plot(w, mag_db, color="steelblue", linewidth=1.5)
    axes[0].set_ylabel("Kazanç (dB)", fontsize=11)
    axes[0].set_title(f"Butterworth Bant Geçiren Filtre Frekans Yanıtı\n"
                      f"(Derece: {bp_order}, Geçiş Bandı: {bp_low}–{bp_high} Hz)",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylim(-80, 5)
    axes[0].axvline(bp_low, color="red", linestyle="--", alpha=0.7, label=f"{bp_low} Hz")
    axes[0].axvline(bp_high, color="red", linestyle="--", alpha=0.7, label=f"{bp_high} Hz")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Phase
    phase_deg = np.rad2deg(np.unwrap(np.angle(h)))
    axes[1].plot(w, phase_deg, color="darkorange", linewidth=1.5)
    axes[1].set_ylabel("Faz (derece)", fontsize=11)
    axes[1].set_xlabel("Frekans (Hz)", fontsize=11)
    axes[1].axvline(bp_low, color="red", linestyle="--", alpha=0.7)
    axes[1].axvline(bp_high, color="red", linestyle="--", alpha=0.7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = figures_dir / "fig_frequency_response.png"
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path2}")


def main():
    parser = argparse.ArgumentParser(description="Clean and filter heart sound segments")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute RMS stats without writing files")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing processed segments")
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
        verify_processed(config, PROJECT_ROOT)
        return

    if args.figures_only:
        generate_figures(config, PROJECT_ROOT)
        return

    clean_and_filter(config, PROJECT_ROOT, dry_run=args.dry_run)
    if not args.dry_run:
        verify_processed(config, PROJECT_ROOT)
        generate_figures(config, PROJECT_ROOT)


if __name__ == "__main__":
    main()
