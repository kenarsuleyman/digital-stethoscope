"""
Heart Sound Dataset Downloader

Downloads PhysioNet/CinC 2016 (training a-e) and PASCAL Dataset-B,
then generates a unified metadata.csv.

Usage:
    python -m src.data.download                    # Download all
    python -m src.data.download --physionet-only   # PhysioNet only
    python -m src.data.download --pascal-only      # PASCAL only
    python -m src.data.download --metadata-only    # Regenerate metadata only
    python -m src.data.download --dry-run          # Show what would be done
    python -m src.data.download --verify           # Verify existing downloads
"""

import argparse
import csv
import logging
import shutil
import time
import zipfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Path = None) -> dict:
    """Load configuration from config.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_file(url: str, dest_path: Path, max_retries: int = 3,
                  connect_timeout: int = 15, read_timeout: int = 30,
                  stall_timeout: int = 60) -> Path:
    """
    Download a file with progress bar, retry logic, stall detection,
    and skip-if-exists.

    - connect_timeout: seconds to wait for initial connection
    - read_timeout: seconds to wait for each chunk read
    - stall_timeout: abort if no data received for this many seconds
    """
    partial_path = dest_path.with_suffix(dest_path.suffix + ".partial")

    # Check if already downloaded
    if dest_path.exists():
        try:
            head = requests.head(url, timeout=connect_timeout, allow_redirects=True)
            remote_size = int(head.headers.get("Content-Length", 0))
            local_size = dest_path.stat().st_size
            if remote_size > 0 and local_size == remote_size:
                logger.info(f"Already downloaded: {dest_path.name} ({local_size / 1e6:.1f} MB)")
                return dest_path
        except requests.RequestException:
            pass

    if partial_path.exists():
        partial_path.unlink()

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading {url} (attempt {attempt}/{max_retries})")
            resp = requests.get(url, stream=True,
                                timeout=(connect_timeout, read_timeout))
            resp.raise_for_status()

            total_size = int(resp.headers.get("Content-Length", 0))
            desc = dest_path.name
            last_data_time = time.time()

            with open(partial_path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True,
                          desc=desc, disable=False) as pbar:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            last_data_time = time.time()
                        elif time.time() - last_data_time > stall_timeout:
                            raise ConnectionError(
                                f"Download stalled — no data for {stall_timeout}s")

            partial_path.rename(dest_path)
            logger.info(f"Downloaded: {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB)")
            return dest_path

        except (requests.RequestException, IOError, ConnectionError) as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if partial_path.exists():
                partial_path.unlink()
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to download {url} after {max_retries} attempts: {e}"
                )


def download_physionet(config: dict, project_root: Path,
                       dry_run: bool = False) -> Path:
    """
    Copy PhysioNet/CinC 2016 training-{a..e} from local path or download zip.

    Prefers local_path from config if it exists. Falls back to downloading
    training.zip from PhysioNet.
    """
    pn_config = config["downloads"]["physionet"]
    pn_dir = project_root / config["paths"]["physionet_dir"]
    pn_dir.mkdir(parents=True, exist_ok=True)

    wanted_sets = pn_config["training_sets"]

    # Check if already in place
    all_present = True
    for ts in wanted_sets:
        ts_dir = pn_dir / ts
        expected = pn_config["expected_counts"].get(ts, 0)
        if not ts_dir.exists() or len(list(ts_dir.glob("*.wav"))) < expected:
            all_present = False
            break

    if all_present:
        logger.info("PhysioNet training sets already in place. Skipping.")
        return pn_dir

    # Try local path first
    local_path = pn_config.get("local_path")
    if local_path:
        local_dir = Path(local_path)
        if local_dir.exists():
            if dry_run:
                logger.info(f"[DRY RUN] Would copy from local: {local_dir}")
                return pn_dir

            logger.info(f"Copying PhysioNet from local path: {local_dir}")
            for ts in wanted_sets:
                src_dir = local_dir / ts
                dst_dir = pn_dir / ts
                if not src_dir.exists():
                    logger.warning(f"Local {ts} not found at {src_dir}")
                    continue
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                logger.info(f"  Copying {ts}...")
                shutil.copytree(src_dir, dst_dir)
                wav_count = len(list(dst_dir.glob("*.wav")))
                expected = pn_config["expected_counts"].get(ts, 0)
                status = "OK" if wav_count >= expected else "MISMATCH"
                logger.info(f"  {ts}: {wav_count} wav files (expected {expected}) [{status}]")

            return pn_dir
        else:
            logger.warning(f"Local path not found: {local_dir}. Falling back to download.")

    # Fallback: download training.zip
    zip_url = pn_config["training_zip_url"]
    zip_path = pn_dir / "training.zip"

    if dry_run:
        logger.info(f"[DRY RUN] Would download: {zip_url}")
        logger.info(f"[DRY RUN] Would extract to: {pn_dir}")
        return pn_dir

    download_file(zip_url, zip_path)

    logger.info("Extracting PhysioNet training sets...")
    wanted_set = set(wanted_sets)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            all_names = zf.namelist()
            first = all_names[0] if all_names else ""
            strip_prefix = "training/" if first.startswith("training/training-") else ""
            wanted_prefixes = tuple(f"{strip_prefix}{s}/" for s in wanted_set)

            members = [m for m in all_names if m.startswith(wanted_prefixes)]
            logger.info(f"Extracting {len(members)} files...")

            for member in tqdm(members, desc="Extracting", unit="file"):
                target_rel = member[len(strip_prefix):] if strip_prefix else member
                target_path = pn_dir / target_rel
                if member.endswith("/"):
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(target_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile:
        logger.error("Corrupt zip file. Deleting — please re-run.")
        zip_path.unlink(missing_ok=True)
        raise

    zip_path.unlink(missing_ok=True)
    logger.info("PhysioNet extraction complete.")

    for ts in wanted_sets:
        ts_dir = pn_dir / ts
        expected = pn_config["expected_counts"].get(ts, 0)
        actual = len(list(ts_dir.glob("*.wav"))) if ts_dir.exists() else 0
        status = "OK" if actual >= expected else "MISMATCH"
        logger.info(f"  {ts}: {actual} wav files (expected {expected}) [{status}]")

    return pn_dir


def download_pascal(config: dict, project_root: Path,
                    dry_run: bool = False) -> Path:
    """
    Download PASCAL Dataset-B (3 zips) and extract.
    """
    ps_config = config["downloads"]["pascal"]
    ps_dir = project_root / config["paths"]["pascal_dir"]
    ps_dir.mkdir(parents=True, exist_ok=True)

    base_url = ps_config["base_url"]

    for file_info in ps_config["files"]:
        zip_name = file_info["name"]
        label = file_info["label"]
        extract_dir = file_info["extract_dir"]
        expected = file_info["expected_count"]
        url = f"{base_url}/{zip_name}"
        zip_path = ps_dir / zip_name
        target_dir = ps_dir / extract_dir

        # Check if already extracted
        if target_dir.exists():
            wav_count = len(list(target_dir.glob("*.wav")))
            if wav_count >= expected:
                logger.info(f"PASCAL {label} already extracted ({wav_count} files). Skipping.")
                continue

        if dry_run:
            logger.info(f"[DRY RUN] Would download: {url}")
            logger.info(f"[DRY RUN] Would extract to: {target_dir}")
            continue

        # Download
        download_file(url, zip_path)

        # Extract
        logger.info(f"Extracting {zip_name}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(ps_dir)
        except zipfile.BadZipFile:
            logger.error(f"Corrupt zip: {zip_name}. Deleting and please re-run.")
            zip_path.unlink(missing_ok=True)
            raise

        # Clean up __MACOSX if present
        macosx_dir = ps_dir / "__MACOSX"
        if macosx_dir.exists():
            shutil.rmtree(macosx_dir)

        # Clean up zip
        zip_path.unlink(missing_ok=True)

        # If files were extracted flat (no subdirectory), move them
        if not target_dir.exists():
            # Files might have been extracted directly into ps_dir
            # Find wav files not in any known subdirectory
            logger.warning(f"Expected dir {target_dir} not found after extraction. "
                           f"Check zip structure manually.")
        else:
            wav_count = len(list(target_dir.glob("*.wav")))
            logger.info(f"  {label}: {wav_count} wav files (expected {expected})")

    return ps_dir


def _parse_physionet_metadata(physionet_dir: Path, config: dict) -> list[dict]:
    """Parse REFERENCE.csv from each PhysioNet training folder."""
    pn_config = config["downloads"]["physionet"]
    label_map = {int(k): v for k, v in pn_config["label_map"].items()}
    binary_map = config["labels"]["binary_map"]
    sr = pn_config["sample_rate"]
    rows = []

    for ts in pn_config["training_sets"]:
        ts_dir = physionet_dir / ts
        ref_file = ts_dir / pn_config["label_file"]

        if not ref_file.exists():
            logger.warning(f"Missing {ref_file}, skipping {ts}")
            continue

        with open(ref_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) < 2:
                    continue
                record_id = line[0].strip()
                label_val = int(line[1].strip())
                original_label = label_map.get(label_val, "Unknown")

                wav_path = ts_dir / f"{record_id}.wav"
                if not wav_path.exists():
                    logger.warning(f"WAV not found: {wav_path}")
                    continue

                rel_path = wav_path.relative_to(PROJECT_ROOT).as_posix()
                rows.append({
                    "file_path": rel_path,
                    "dataset_source": "physionet",
                    "original_sr": sr,
                    "original_label": original_label,
                    "binary_label": binary_map.get(original_label, -1),
                    "patient_id": record_id,
                })

    return rows


def _parse_pascal_metadata(pascal_dir: Path, config: dict) -> list[dict]:
    """Build metadata from PASCAL directory structure."""
    ps_config = config["downloads"]["pascal"]
    binary_map = config["labels"]["binary_map"]
    sr = ps_config["sample_rate"]
    rows = []

    for file_info in ps_config["files"]:
        label = file_info["label"]
        extract_dir = file_info["extract_dir"]
        target_dir = pascal_dir / extract_dir

        if not target_dir.exists():
            logger.warning(f"Missing {target_dir}, skipping {label}")
            continue

        for wav_file in sorted(target_dir.glob("*.wav")):
            rel_path = wav_file.relative_to(PROJECT_ROOT).as_posix()
            patient_id = wav_file.stem
            rows.append({
                "file_path": rel_path,
                "dataset_source": "pascal",
                "original_sr": sr,
                "original_label": label,
                "binary_label": binary_map.get(label, -1),
                "patient_id": patient_id,
            })

    return rows


def generate_metadata(config: dict, project_root: Path) -> Path:
    """Combine PhysioNet and PASCAL metadata into a single CSV."""
    physionet_dir = project_root / config["paths"]["physionet_dir"]
    pascal_dir = project_root / config["paths"]["pascal_dir"]
    metadata_path = project_root / config["paths"]["metadata_csv"]

    logger.info("Generating metadata...")
    pn_rows = _parse_physionet_metadata(physionet_dir, config)
    ps_rows = _parse_pascal_metadata(pascal_dir, config)

    all_rows = sorted(pn_rows + ps_rows, key=lambda r: (r["dataset_source"], r["file_path"]))

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_path", "dataset_source", "original_sr", "original_label",
                  "binary_label", "patient_id"]

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Print summary
    pn_count = len(pn_rows)
    ps_count = len(ps_rows)
    total = len(all_rows)
    normal = sum(1 for r in all_rows if r["binary_label"] == 0)
    abnormal = sum(1 for r in all_rows if r["binary_label"] == 1)

    logger.info(f"Metadata written to {metadata_path}")
    logger.info(f"  PhysioNet: {pn_count} recordings")
    logger.info(f"  PASCAL:    {ps_count} recordings")
    logger.info(f"  Total:     {total} recordings")
    logger.info(f"  Normal:    {normal}  |  Abnormal: {abnormal}")

    return metadata_path


def verify_downloads(config: dict, project_root: Path) -> bool:
    """Verify download integrity: file counts, WAV headers, metadata."""
    all_ok = True
    logger.info("=" * 50)
    logger.info("VERIFICATION REPORT")
    logger.info("=" * 50)

    # PhysioNet
    pn_config = config["downloads"]["physionet"]
    pn_dir = project_root / config["paths"]["physionet_dir"]
    pn_total = 0

    for ts in pn_config["training_sets"]:
        ts_dir = pn_dir / ts
        expected = pn_config["expected_counts"].get(ts, 0)
        actual = len(list(ts_dir.glob("*.wav"))) if ts_dir.exists() else 0
        pn_total += actual
        ok = actual >= expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        logger.info(f"  [{status}] {ts}: {actual}/{expected} wav files")

    # PASCAL
    ps_config = config["downloads"]["pascal"]
    ps_dir = project_root / config["paths"]["pascal_dir"]
    ps_total = 0

    for file_info in ps_config["files"]:
        extract_dir = file_info["extract_dir"]
        expected = file_info["expected_count"]
        target_dir = ps_dir / extract_dir
        actual = len(list(target_dir.glob("*.wav"))) if target_dir.exists() else 0
        ps_total += actual
        ok = actual >= expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        logger.info(f"  [{status}] {extract_dir}: {actual}/{expected} wav files")

    # Metadata
    metadata_path = project_root / config["paths"]["metadata_csv"]
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            meta_count = sum(1 for _ in reader)
        expected_total = pn_total + ps_total
        ok = meta_count == expected_total
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        logger.info(f"  [{status}] metadata.csv: {meta_count} rows (expected {expected_total})")
    else:
        logger.info("  [FAIL] metadata.csv not found")
        all_ok = False

    # WAV header spot check
    import random
    all_wavs = list(pn_dir.rglob("*.wav")) + list(ps_dir.rglob("*.wav"))
    sample_size = min(10, len(all_wavs))
    if sample_size > 0:
        samples = random.sample(all_wavs, sample_size)
        bad = 0
        for wav in samples:
            with open(wav, "rb") as f:
                header = f.read(4)
                if header != b"RIFF":
                    bad += 1
                    logger.warning(f"  Bad WAV header: {wav}")
        status = "PASS" if bad == 0 else "FAIL"
        if bad > 0:
            all_ok = False
        logger.info(f"  [{status}] WAV header check: {sample_size - bad}/{sample_size} valid")

    logger.info("=" * 50)
    overall = "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"
    logger.info(f"  {overall}")
    logger.info("=" * 50)

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Download heart sound datasets")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--physionet-only", action="store_true",
                        help="Download PhysioNet only")
    parser.add_argument("--pascal-only", action="store_true",
                        help="Download PASCAL only")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only regenerate metadata.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without downloading")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing downloads")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)

    if args.verify:
        verify_downloads(config, PROJECT_ROOT)
        return

    if not args.metadata_only:
        if not args.pascal_only:
            download_physionet(config, PROJECT_ROOT, dry_run=args.dry_run)
        if not args.physionet_only:
            download_pascal(config, PROJECT_ROOT, dry_run=args.dry_run)

    if not args.dry_run:
        generate_metadata(config, PROJECT_ROOT)
        verify_downloads(config, PROJECT_ROOT)


if __name__ == "__main__":
    main()
