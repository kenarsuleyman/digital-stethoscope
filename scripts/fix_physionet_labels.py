"""
Rewrite the PhysioNet/CinC 2016 labels in every pipeline CSV from the challenge's own REFERENCE files.

Background: configs/config.yaml mapped REFERENCE 1 -> "Normal" and -1 -> "Abnormal". PhysioNet 2016 uses
1 = abnormal and -1 = normal (the .hea headers say "# Abnormal" / "# Normal"), so every PhysioNet label in
metadata.csv and the downstream CSVs was inverted. This script sets them from REFERENCE.csv directly, so it is
idempotent (running it twice changes nothing). Mel .npy files do not contain labels and are not touched.
Patient groups are unchanged, so split membership stays the same; class weights are recomputed from train.csv
exactly as src/data/split.py does.

Usage:
    python scripts/fix_physionet_labels.py            # dry run: report what would change
    python scripts/fix_physionet_labels.py --apply    # rewrite the CSVs and configs/class_weights.json
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
CSVS = ["data/metadata.csv", "data/segments_metadata.csv", "data/processed_metadata.csv",
        "data/mel_metadata.csv", "splits/train.csv", "splits/val.csv", "splits/test.csv"]


def reference_labels(config: dict) -> dict:
    """record id -> 1 (abnormal) / 0 (normal), checked against the .hea header of every record."""
    pn = config["downloads"]["physionet"]
    root = ROOT / config["paths"]["physionet_dir"]
    labels = {}
    for subset in pn["training_sets"]:
        d = root / subset
        with open(d / pn["label_file"], newline="") as f:
            for rec, val in csv.reader(f):
                lab = 1 if int(val) == 1 else 0
                hea = (d / f"{rec}.hea").read_text(encoding="utf-8", errors="ignore")
                expected = "# Abnormal" if lab else "# Normal"
                if expected not in hea:
                    raise SystemExit(f"{subset}/{rec}: REFERENCE={val} but header lacks '{expected}'")
                labels[rec] = lab
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    config = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
    pn_map = {int(k): v for k, v in config["downloads"]["physionet"]["label_map"].items()}
    if pn_map != {1: "Abnormal", -1: "Normal"}:
        raise SystemExit(f"Fix configs/config.yaml first: physionet label_map is {pn_map}")

    ref = reference_labels(config)
    print(f"PhysioNet REFERENCE: {len(ref)} records, {sum(ref.values())} abnormal / "
          f"{len(ref) - sum(ref.values())} normal")

    for rel in CSVS:
        path = ROOT / rel
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            fields = list(rows[0].keys())
        changed, pn_rows, after = 0, 0, Counter()
        for r in rows:
            lab = ref.get(r["patient_id"])
            if lab is None:
                continue
            if "dataset_source" in r and r["dataset_source"] != "physionet":
                raise SystemExit(f"{rel}: patient_id {r['patient_id']} is not a PhysioNet row")
            pn_rows += 1
            if int(r["binary_label"]) != lab:
                changed += 1
                r["binary_label"] = str(lab)
            if "original_label" in r:
                r["original_label"] = "Abnormal" if lab else "Normal"
            after[lab] += 1
        print(f"{rel:28s} PhysioNet rows {pn_rows:6d}  labels changed {changed:6d}  "
              f"-> normal {after[0]:6d} / abnormal {after[1]:6d}")
        if args.apply and changed:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
                w.writeheader()
                w.writerows(rows)

    # class weights from train.csv, as in src/data/split.py
    with open(ROOT / "splits" / "train.csv", newline="", encoding="utf-8") as f:
        y = np.array([int(r["binary_label"]) for r in csv.DictReader(f)])
    if not args.apply:
        # dry run: apply the PhysioNet fix in memory
        with open(ROOT / "splits" / "train.csv", newline="", encoding="utf-8") as f:
            y = np.array([ref.get(r["patient_id"], int(r["binary_label"])) for r in csv.DictReader(f)])
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"train.csv: normal {int((y == 0).sum())} / abnormal {int((y == 1).sum())}  class weights {cw}")
    if args.apply:
        with open(ROOT / "configs" / "class_weights.json", "w") as f:
            json.dump(cw, f, indent=2)
        print("written. Next: python scripts/package_for_colab.py --skip-zip, then upload src/, configs/, splits/")
    else:
        print("dry run only (use --apply to write)")


if __name__ == "__main__":
    main()
