"""
One-shot helper: build a small WAV parity set + Python "golden" predictions,
then copy the WAVs into the React Native project for on-device parity testing.

Output:
  test_parity/
    wavs/                    -- the chosen WAV files (also copied to RN project)
    golden.json              -- per-file Python prediction (probability, label,
                                per-segment probs, source SR, etc.)

The same WAV folder is mirrored to the RN project so Claude Code in that project
can load and predict each file, then diff against golden.json.
"""

import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

from src.inference.predict_wav import classify_file, load_backend
from src.preprocessing.clean_filter import load_config

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = Path("C:/Users/kenar/Desktop/digital-stethoscope-rn/android/app/src/main/assets/heart_sound_classifier_fp16.tflite")
RN_TEST_DIR = Path("C:/Users/kenar/Desktop/digital-stethoscope-rn/test_wavs")
EXTERNAL_DIR = Path("C:/Users/kenar/Desktop/heart-sounds-public")
LOCAL_OUT = PROJECT_ROOT / "test_parity"

# Pick N segments per (dataset_prefix, label) pair to get diverse mix
SEG_CSV = PROJECT_ROOT / "data" / "segments_metadata.csv"
TEST_CSV = PROJECT_ROOT / "splits" / "test.csv"

PER_GROUP = 2          # 2 segments per (dataset, label) -> ~16-20 files
EXTERNAL_FILES = [     # known held-out external clips
    "Normal.wav",
    "Aortic-Stenosis.wav",
    "Aortic-Regurgitation.wav",
    "Mitral-Regurgitation.wav",
    "Mitral-Stenosis.wav",
    "Pulmonary-stenosis.wav",
]


def dataset_prefix(patient_id: str) -> str:
    for p in ("physionet", "pascal", "yaseen", "buet", "circor"):
        if patient_id.startswith(p):
            return p
    # PhysioNet patient IDs often look like "a0001" -> "training-a"
    if patient_id and patient_id[0] in "abcde":
        return "physionet"
    return "other"


def pick_test_segments() -> list[dict]:
    """Choose a balanced sample of WAV segments from test patients."""
    import csv

    test_pids = set()
    test_labels = {}
    with open(TEST_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            test_pids.add(row["patient_id"])
            test_labels[row["patient_id"]] = int(row["binary_label"])

    # group segments by (dataset, label)
    buckets = defaultdict(list)
    with open(SEG_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["patient_id"]
            if pid not in test_pids:
                continue
            seg_path = PROJECT_ROOT / row["segment_path"]
            if not seg_path.exists():
                continue
            label = int(row["binary_label"])
            ds = dataset_prefix(pid)
            buckets[(ds, label)].append({
                "wav": seg_path,
                "patient_id": pid,
                "label": label,
                "source": ds,
                "origin": "test_split",
            })

    rng = random.Random(42)
    picks = []
    for key, items in sorted(buckets.items()):
        rng.shuffle(items)
        picks.extend(items[:PER_GROUP])
    return picks


def pick_external() -> list[dict]:
    """Add never-seen external clips."""
    out = []
    for name in EXTERNAL_FILES:
        p = EXTERNAL_DIR / name
        if not p.exists():
            logger.warning(f"missing external: {p}")
            continue
        label = 0 if "normal" in name.lower() else 1
        out.append({
            "wav": p,
            "patient_id": f"external_{p.stem}",
            "label": label,
            "source": "heart-sounds-public",
            "origin": "external_holdout",
        })
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(None)
    backend = load_backend(MODEL_PATH)

    picks = pick_test_segments() + pick_external()
    logger.info(f"Selected {len(picks)} WAVs")

    LOCAL_OUT.mkdir(exist_ok=True)
    (LOCAL_OUT / "wavs").mkdir(exist_ok=True)
    RN_TEST_DIR.mkdir(exist_ok=True)

    golden = []
    for i, item in enumerate(picks, 1):
        src: Path = item["wav"]
        # canonical filename: <origin>__<source>__<label>__<basename>
        truth = "normal" if item["label"] == 0 else "abnormal"
        # keep RN filename simple ASCII (yaseen Korean dirs would break that path)
        safe = src.name.replace(" ", "_")
        new_name = f"{i:02d}_{item['source']}_{truth}_{safe}"

        dst_local = LOCAL_OUT / "wavs" / new_name
        dst_rn = RN_TEST_DIR / new_name
        shutil.copy2(src, dst_local)
        shutil.copy2(src, dst_rn)

        res = classify_file(src, backend, config, threshold=0.5, apply_rms_gate=False)
        if res is None:
            logger.warning(f"  skipped (too short): {src.name}")
            continue
        entry = {
            "rn_filename": new_name,
            "source": item["source"],
            "origin": item["origin"],
            "patient_id": item["patient_id"],
            "truth_label": item["label"],
            "truth_class": truth,
            "src_sample_rate": res["src_sr"],
            "n_segments": res["n_seg"],
            "python_per_segment": [round(p, 4) for p in res["per_segment"].tolist()],
            "python_mean_prob": round(res["mean_prob"], 4),
            "python_class": res["label"].lower(),
            "correct": (res["label"].lower() == truth),
        }
        golden.append(entry)
        logger.info(f"  {new_name}  truth={truth}  python={res['label']:<8}  "
                    f"p={res['mean_prob']:.3f}  n_seg={res['n_seg']}")

    summary = {
        "model_path": str(MODEL_PATH),
        "threshold": 0.5,
        "n_files": len(golden),
        "agreement_with_truth": sum(1 for g in golden if g["correct"]) / max(len(golden), 1),
        "items": golden,
    }
    out_json = LOCAL_OUT / "golden.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # mirror golden into RN side too, so the on-device test script can read it
    (RN_TEST_DIR / "golden.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("")
    logger.info(f"Wrote {out_json}")
    logger.info(f"Mirrored WAVs + golden.json to {RN_TEST_DIR}")
    logger.info(f"Agreement with ground truth (Python tflite): "
                f"{summary['agreement_with_truth'] * 100:.1f}%")


if __name__ == "__main__":
    main()
