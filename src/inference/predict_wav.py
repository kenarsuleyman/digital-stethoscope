"""
Golden-reference inference — runs the EXACT training pipeline on any raw WAV.

This is the diagnostic instrument for the train/serve mismatch. It reuses the
real preprocessing functions from the training pipeline (no reimplementation),
so its output is what the model "should" say. Compare it against the React app:

  - Python says Normal  but React says Abnormal  -> serving skew (fix the JS).
  - Python ALSO says Abnormal for a healthy signal -> model bias / domain shift
    (retrain / recalibrate threshold / add domain data), NOT a React bug.

Pipeline (identical to stages 2-4): resample->4000 -> 5 s segments ->
Butterworth 25-900 Hz (sosfiltfilt) -> normalize [-1,1] -> log-mel [0,1] ->
224x224x3. Then the model produces a per-segment sigmoid probability; the file
verdict is the mean over segments at threshold 0.5 (same as the React app).

Note on the yaseen-samples folder: the *_repeat3 / *_concat files are just the
short originals repeated/concatenated to lengthen them. For a clean batch test
point --folder at the subfolders that hold the originals (N_New_* = Normal,
AS/MR/MS/MVP_New_* = Abnormal) with --recursive.

Usage:
    # single recording (your own / a yaseen file), using the deployed tflite
    python -m src.inference.predict_wav --model PATH/heart_sound_classifier_fp16.tflite \
        --wav "C:/Users/kenar/Desktop/yaseen-samples/N_New_3주기/New_N_001.wav"

    # a Keras model instead (needs TensorFlow)
    python -m src.inference.predict_wav --model output/models/best_model.keras --wav rec.wav

    # batch-evaluate a labelled folder -> confusion matrix + sensitivity/specificity
    python -m src.inference.predict_wav --model model.tflite \
        --folder "C:/Users/kenar/Desktop/yaseen-samples" --recursive

    # dump the 224x224x3 mel of the first segment for numeric parity vs the JS app
    python -m src.inference.predict_wav --wav rec.wav --dump-mel mel_dump.npy --no-model
"""

import argparse
import logging
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, sosfiltfilt

# Reuse the REAL training-pipeline functions so preprocessing is identical.
from src.preprocessing.clean_filter import design_bandpass, compute_rms, load_config
from src.preprocessing.mel_spectrogram import audio_to_mel_image

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Paths containing one of these (case-insensitive) are treated as Normal ground
# truth in --folder mode; everything else counts as Abnormal.
NORMAL_KEYWORDS = ("normal", "n_new", "_n_", "/n_", "\\n_")


# --------------------------------------------------------------------------- #
# Preprocessing (mirrors stages 2-4, using the imported pipeline functions)
# --------------------------------------------------------------------------- #
def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float64")
    if audio.ndim > 1:  # stereo -> mono
        audio = audio.mean(axis=1)
    return audio, sr


def resample_to_target(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    frac = Fraction(target_sr, sr).limit_denominator(1000)
    return resample_poly(audio, up=frac.numerator, down=frac.denominator)


def segment_audio(audio: np.ndarray, seg_samples: int) -> list[np.ndarray]:
    n = len(audio) // seg_samples
    return [audio[i * seg_samples:(i + 1) * seg_samples] for i in range(n)]


def preprocess_segment(seg: np.ndarray, sos, mel_config: dict,
                       target_sr: int, image_size: tuple) -> np.ndarray:
    """Bandpass -> normalize [-1,1] -> log-mel [0,1] 224x224x3 (== stages 3-4)."""
    filtered = sosfiltfilt(sos, seg)
    max_abs = np.max(np.abs(filtered))
    if max_abs > 0:
        filtered = filtered / max_abs
    return audio_to_mel_image(filtered, target_sr, mel_config, image_size)


def recording_to_mels(path: Path, config: dict, apply_rms_gate: bool):
    """Return (mel images, segment RMS list, kept mask, source sample rate)."""
    prep = config["preprocessing"]
    mel_config = config["mel_spectrogram"]
    target_sr = prep["target_sr"]
    seg_samples = target_sr * prep["segment_length_s"]
    image_size = tuple(mel_config["image_size"])

    audio, sr = load_audio_mono(path)
    audio = resample_to_target(audio, sr, target_sr)
    segments = segment_audio(audio, seg_samples)
    if not segments:
        return [], [], [], sr

    rms_vals = [compute_rms(s) for s in segments]

    # RMS gate, replicating the React app (per-recording median). The training
    # pipeline used the GLOBAL dataset median instead -- this is a known
    # difference, so gating is OFF by default for single-recording inference.
    keep = [True] * len(segments)
    if apply_rms_gate:
        med = float(np.median([r for r in rms_vals if r > 0]) or 0.0)
        if med > 0:
            lo = prep["rms_low_factor"] * med
            hi = prep["rms_high_factor"] * med
            keep = [lo <= r <= hi for r in rms_vals]

    sos = design_bandpass(prep["bandpass_low"], prep["bandpass_high"],
                          target_sr, prep["butterworth_order"])
    mels = [preprocess_segment(s, sos, mel_config, target_sr, image_size)
            for s, k in zip(segments, keep) if k]
    return mels, rms_vals, keep, sr


# --------------------------------------------------------------------------- #
# Model backends (tflite preferred -- it's what the React app runs)
# --------------------------------------------------------------------------- #
class TFLiteModel:
    def __init__(self, path: Path):
        Interpreter = _import_tflite_interpreter()
        self.interp = Interpreter(model_path=str(path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def predict_batch(self, mels: list[np.ndarray]) -> np.ndarray:
        probs = []
        for mel in mels:
            x = mel[None, ...].astype(self.inp["dtype"])
            self.interp.set_tensor(self.inp["index"], x)
            self.interp.invoke()
            probs.append(float(self.interp.get_tensor(self.out["index"]).reshape(-1)[0]))
        return np.array(probs)


class KerasModel:
    def __init__(self, path: Path):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(str(path))

    def predict_batch(self, mels: list[np.ndarray]) -> np.ndarray:
        x = np.stack(mels).astype("float32")
        return self.model.predict(x, verbose=0).reshape(-1)


def _import_tflite_interpreter():
    for mod, attr in (("ai_edge_litert.interpreter", "Interpreter"),
                      ("tensorflow.lite", "Interpreter"),
                      ("tflite_runtime.interpreter", "Interpreter")):
        try:
            m = __import__(mod, fromlist=[attr])
            return getattr(m, attr)
        except ImportError:
            continue
    raise ImportError(
        "No TFLite runtime found. Install one of:\n"
        "    pip install ai-edge-litert    (lightweight, recommended)\n"
        "    pip install tensorflow        (also runs .keras models)")


def load_backend(model_path: Path):
    if model_path.suffix.lower() in (".keras", ".h5"):
        return KerasModel(model_path)
    if model_path.suffix.lower() == ".tflite":
        return TFLiteModel(model_path)
    raise ValueError(f"Unknown model type: {model_path.suffix}")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def classify_file(path: Path, backend, config: dict, threshold: float,
                  apply_rms_gate: bool) -> dict | None:
    mels, rms_vals, keep, sr = recording_to_mels(path, config, apply_rms_gate)
    if not mels:
        logger.warning(f"  {path.name}: too short (<5 s after resample) — skipped")
        return None
    probs = backend.predict_batch(mels)
    mean_p = float(probs.mean())
    return {
        "file": path.name, "src_sr": sr, "n_seg": len(probs),
        "per_segment": probs, "mean_prob": mean_p,
        "label": "Abnormal" if mean_p >= threshold else "Normal",
    }


def label_from_path(path: Path) -> int:
    s = str(path).lower()
    return 0 if any(k in s for k in NORMAL_KEYWORDS) else 1


def run_single(args, backend, config):
    res = classify_file(Path(args.wav), backend, config, args.threshold,
                        args.rms_gate)
    if res is None:
        return
    print("\n" + "=" * 60)
    print(f"  File:        {res['file']}  (src {res['src_sr']} Hz)")
    print(f"  Segments:    {res['n_seg']}  (5 s each)")
    seg_str = ", ".join(f"{p:.3f}" for p in res["per_segment"])
    print(f"  Per-segment: [{seg_str}]")
    print(f"  Mean prob:   {res['mean_prob']:.4f}   (1=Abnormal, 0=Normal)")
    print(f"  VERDICT:     {res['label']}   @ threshold {args.threshold}")
    print("=" * 60)


def run_folder(args, backend, config):
    folder = Path(args.folder)
    pat = "**/*.wav" if args.recursive else "*.wav"
    files = sorted(folder.glob(pat))
    if not files:
        logger.error(f"No .wav files under {folder}")
        return

    rows, y_true, y_pred = [], [], []
    for f in files:
        res = classify_file(f, backend, config, args.threshold, args.rms_gate)
        if res is None:
            continue
        gt = label_from_path(f)
        res["truth"] = "Normal" if gt == 0 else "Abnormal"
        rows.append(res)
        y_true.append(gt)
        y_pred.append(1 if res["mean_prob"] >= args.threshold else 0)

    print(f"\n{'file':<40} {'truth':<9} {'pred':<9} {'mean_p':>7}")
    print("-" * 70)
    for r in rows:
        mark = "" if r["truth"] == r["label"] else "  <-- MISS"
        print(f"{r['file'][:39]:<40} {r['truth']:<9} {r['label']:<9} "
              f"{r['mean_prob']:>7.3f}{mark}")

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = len(y_true)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / n if n else float("nan")

    print("\n" + "=" * 60)
    print("  CONFUSION MATRIX                pred:Normal  pred:Abnormal")
    print(f"  truth:Normal                    {tn:>10}   {fp:>12}")
    print(f"  truth:Abnormal                  {fn:>10}   {tp:>12}")
    print("-" * 60)
    print(f"  Accuracy:    {acc:.3f}   ({tp + tn}/{n})")
    print(f"  Sensitivity: {sens:.3f}   (abnormal correctly flagged)")
    print(f"  Specificity: {spec:.3f}   (normal correctly cleared)  <-- your problem area")
    print("=" * 60)


def run_dump_mel(args, config):
    mels, _, _, sr = recording_to_mels(Path(args.wav), config, args.rms_gate)
    if not mels:
        logger.error("No segments produced — recording too short.")
        return
    out = Path(args.dump_mel)
    np.save(out, mels[0])
    print(f"Saved first-segment mel {mels[0].shape} -> {out}")
    print(f"  min={mels[0].min():.4f} max={mels[0].max():.4f} mean={mels[0].mean():.4f}")
    print("  Compare these stats (and the array) against the JS app's tensor")
    print("  for the SAME wav to localise the preprocessing skew.")


def main():
    p = argparse.ArgumentParser(description="Golden-reference heart-sound inference")
    p.add_argument("--model", type=Path, help="Path to .tflite or .keras model")
    p.add_argument("--wav", type=str, help="Single WAV to classify")
    p.add_argument("--folder", type=str, help="Folder of WAVs to batch-evaluate")
    p.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--rms-gate", action="store_true",
                   help="Apply per-recording RMS gating (as the React app does)")
    p.add_argument("--dump-mel", type=str, help="Save first-segment mel to this .npy")
    p.add_argument("--no-model", action="store_true",
                   help="Skip model load (use with --dump-mel for preprocessing-only)")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(args.config)

    if args.dump_mel and args.no_model:
        run_dump_mel(args, config)
        return

    if not args.model:
        p.error("--model is required (or use --dump-mel --no-model)")
    backend = load_backend(args.model)

    if args.dump_mel:
        run_dump_mel(args, config)
    if args.folder:
        run_folder(args, backend, config)
    elif args.wav:
        run_single(args, backend, config)
    else:
        p.error("Provide --wav or --folder")


if __name__ == "__main__":
    main()
