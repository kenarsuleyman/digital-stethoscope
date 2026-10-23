"""
Package project files for Google Drive / Colab upload.

Creates a folder structure ready to upload to Google Drive:
    drive_upload/
        data.zip          — mel/ .npy files (compressed)
        src/              — source code
        configs/          — config.yaml, class_weights.json
        splits/           — train.csv, val.csv, test.csv

Upload the entire drive_upload/ contents to:
    Google Drive > My Drive > heart-sound-classifier/

Usage:
    python scripts/package_for_colab.py
    python scripts/package_for_colab.py --output-dir drive_upload
    python scripts/package_for_colab.py --skip-zip   # only copy src/configs/splits
"""

import argparse
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def package(output_dir: Path, skip_zip: bool = False, zip_path: Path = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    mel_dir = PROJECT_ROOT / "data" / "mel"
    splits_dir = PROJECT_ROOT / "splits"
    configs_dir = PROJECT_ROOT / "configs"
    src_dir = PROJECT_ROOT / "src"

    # 1. Copy src/ (only .py files, skip __pycache__)
    dst_src = output_dir / "src"
    if dst_src.exists():
        shutil.rmtree(dst_src)
    shutil.copytree(
        src_dir, dst_src,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    print(f"[OK] src/ -> {dst_src}")

    # 2. Copy configs/
    dst_configs = output_dir / "configs"
    if dst_configs.exists():
        shutil.rmtree(dst_configs)
    shutil.copytree(configs_dir, dst_configs)
    print(f"[OK] configs/ -> {dst_configs}")

    # 3. Copy splits/
    dst_splits = output_dir / "splits"
    if dst_splits.exists():
        shutil.rmtree(dst_splits)
    shutil.copytree(splits_dir, dst_splits)
    print(f"[OK] splits/ -> {dst_splits}")

    # 4. Zip mel/ .npy files
    if skip_zip:
        print("[SKIP] data.zip (--skip-zip)")
        return

    npy_files = sorted(mel_dir.glob("*.npy"))
    if not npy_files:
        print("[ERROR] No .npy files found in data/mel/")
        return

    zip_path = zip_path or output_dir / "data.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Zipping {len(npy_files)} mel files -> {zip_path}")
    print("This may take several minutes (~17 GB of .npy -> ~5 GB zip)...")

    # Deflate: the 3 channels of every Mel image are identical copies, so the zip shrinks to ~30 %
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for i, npy in enumerate(npy_files):
            # Store as mel/<filename>.npy so unzip creates mel/ directory
            arcname = f"mel/{npy.name}"
            zf.write(npy, arcname)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(npy_files)} files...")

    size_gb = zip_path.stat().st_size / (1024 ** 3)
    print(f"[OK] data.zip ({size_gb:.2f} GB, {len(npy_files)} files)")

    print()
    print("=" * 55)
    print("PAKETLEME TAMAMLANDI")
    print("=" * 55)
    print()
    print("Sonraki adımlar:")
    print(f"  1. '{output_dir}' klasörünün içeriğini Google Drive'a yükleyin:")
    print(f"     Google Drive > My Drive > heart-sound-classifier/")
    print()
    print("  Drive yapisi soyle olmali:")
    print("     heart-sound-classifier/")
    print("       +-- data.zip")
    print("       +-- src/")
    print("       +-- configs/")
    print("       +-- splits/")
    print()
    print("  2. Google Colab'da notebooks/train_colab.ipynb açın")
    print("  3. Runtime > Change runtime type > T4 GPU seçin")
    print("  4. Tüm hücreleri çalıştırın")


def main():
    parser = argparse.ArgumentParser(description="Package files for Colab")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "drive_upload",
                        help="Output directory (default: drive_upload/)")
    parser.add_argument("--skip-zip", action="store_true",
                        help="Skip creating data.zip (only copy src/configs/splits)")
    parser.add_argument("--zip-path", type=Path, default=None,
                        help="Where to write data.zip (default: <output-dir>/data.zip). "
                             "Use a folder outside OneDrive to avoid syncing a multi-GB file.")
    args = parser.parse_args()

    package(args.output_dir, args.skip_zip, args.zip_path)


if __name__ == "__main__":
    main()
