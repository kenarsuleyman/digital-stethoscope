"""Rebuild the on-device parity set for the retrained model (v2, PhysioNet labels fixed).

Keeps the same 26 WAVs as test_parity/ (make_parity_set.py, built with the v1 model), recomputes the Python
"golden" predictions with the v2 TFLite model and takes the PhysioNet truth labels from the fixed
data/metadata.csv (the four PhysioNet files had inverted labels in test_parity/golden.json). Files whose class
word in the name changes are renamed. test_parity/ is left untouched.

Output: test_parity_v2/wavs/*.wav + test_parity_v2/golden.json (same format as before).
Usage (from the project root):  python scripts/make_golden_v2.py --model <v2 .tflite (ASCII path)>
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference.predict_wav import classify_file, load_backend  # noqa: E402
from src.preprocessing.clean_filter import load_config  # noqa: E402

OLD = ROOT / "test_parity"
NEW = ROOT / "test_parity_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    a = ap.parse_args()

    meta = {r["patient_id"]: int(r["binary_label"])
            for r in csv.DictReader(open(ROOT / "data" / "metadata.csv", encoding="utf-8"))
            if r["dataset_source"] == "physionet"}
    old = json.load(open(OLD / "golden.json", encoding="utf-8"))
    config = load_config(None)
    backend = load_backend(a.model)
    (NEW / "wavs").mkdir(parents=True, exist_ok=True)

    items = []
    for it in old["items"]:
        label = meta[it["patient_id"]] if it["source"] == "physionet" else it["truth_label"]
        truth = "normal" if label == 0 else "abnormal"
        name = it["rn_filename"]
        if label != it["truth_label"]:          # rename "<nn>_<source>_<class>_..." to the true class
            parts = name.split("_", 3)
            name = "_".join([parts[0], parts[1], truth, parts[3]])
        shutil.copy2(OLD / "wavs" / it["rn_filename"], NEW / "wavs" / name)
        res = classify_file(NEW / "wavs" / name, backend, config, threshold=0.5, apply_rms_gate=False)
        e = dict(it)
        e.update({"rn_filename": name, "truth_label": label, "truth_class": truth,
                  "python_per_segment": [round(p, 4) for p in res["per_segment"].tolist()],
                  "python_mean_prob": round(res["mean_prob"], 4), "python_class": res["label"].lower(),
                  "correct": res["label"].lower() == truth, "label_fixed": label != it["truth_label"]})
        items.append(e)
        print(f'{name[:58]:58s} truth={truth:8s} py={e["python_mean_prob"]:.3f} '
              f'(v1 {it["python_mean_prob"]:.3f}){"  LABEL FIXED" if e["label_fixed"] else ""}')

    out = {"model_path": str(a.model), "model_sha1": hashlib.sha1(a.model.read_bytes()).hexdigest(),
           "threshold": 0.5, "n_files": len(items),
           "agreement_with_truth": sum(i["correct"] for i in items) / len(items), "items": items}
    (NEW / "golden.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f'truth agreement {sum(i["correct"] for i in items)}/{len(items)} -> {NEW / "golden.json"}')


if __name__ == "__main__":
    main()
