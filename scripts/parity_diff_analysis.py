"""Where do the desktop (training) and mobile-app preprocessing differ?

The app (digital-stethoscope-mobile, src/lib/audio) re-implements the Python pipeline in TypeScript with
three different conventions:
  F  band-pass: plain forward-backward biquad pass, zero initial state, no edge padding
     (Python: scipy.signal.sosfiltfilt, odd padding + steady-state initial conditions)
  P  STFT centering pad: reflect (Python: librosa 1.0 default 'constant', i.e. zeros)
  R  224x224 resize: bilinear with align_corners=False, half_pixel_centers=False (tfjs default)
     (Python: PIL Image.BILINEAR, half-pixel centers)
Each convention is switched on alone and in combination; the abnormal probability of every parity file is
computed with the desktop TFLite model and compared with the Python golden and the on-device result.

Usage: python scripts/parity_diff_analysis.py --model <v2.tflite> [--set test_parity_v2]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import librosa
import numpy as np
from PIL import Image
from scipy.signal import sosfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference.predict_wav import (load_audio_mono, load_backend, resample_to_target,  # noqa: E402
                                       segment_audio)
from src.preprocessing.clean_filter import load_config  # noqa: E402
from src.inference import predict_wav as pw  # noqa: E402


def filt(x, sos, app):
    if not app:
        from scipy.signal import sosfiltfilt
        return sosfiltfilt(sos, x)
    y = sosfilt(sos, x)                       # zero initial state, no padding (filter.ts)
    return sosfilt(sos, y[::-1])[::-1]


def resize_tfjs(S, size):                     # pipeline.ts resizeBilinearTo3ch
    h, w = S.shape
    ys, xs = np.arange(size) * (h / size), np.arange(size) * (w / size)
    y0 = np.minimum(np.floor(ys).astype(int), h - 1); y1 = np.minimum(y0 + 1, h - 1); wy = ys - y0
    x0 = np.minimum(np.floor(xs).astype(int), w - 1); x1 = np.minimum(x0 + 1, w - 1); wx = xs - x0
    top = S[np.ix_(y0, x0)] + (S[np.ix_(y0, x1)] - S[np.ix_(y0, x0)]) * wx
    bot = S[np.ix_(y1, x0)] + (S[np.ix_(y1, x1)] - S[np.ix_(y1, x0)]) * wx
    return (top + (bot - top) * wy[:, None]).astype(np.float32)


def image(seg, sos, mc, sr, size, F, P, R):
    x = filt(seg, sos, F)
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=mc["n_fft"], hop_length=mc["hop_length"],
                                       n_mels=mc["n_mels"], fmin=mc["fmin"], fmax=mc["fmax"],
                                       pad_mode="reflect" if P else "constant")
    S = librosa.power_to_db(S, ref=np.max)
    S = ((S - S.min()) / (S.max() - S.min())).astype(np.float32) if S.max() > S.min() else np.zeros_like(S, np.float32)
    S = resize_tfjs(S, size[0]) if R else np.array(Image.fromarray(S).resize((size[1], size[0]), Image.BILINEAR),
                                                  dtype=np.float32)
    return np.stack([S, S, S], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--set", default="test_parity_v2")
    a = ap.parse_args()
    d = ROOT / a.set
    golden = json.load(open(d / "golden.json", encoding="utf-8"))
    rn = {x["file"]: x for x in json.load(open(d / "rn_results.json", encoding="utf-8"))["parity"]["per_file"]}
    cfg = load_config(None); prep, mc = cfg["preprocessing"], cfg["mel_spectrogram"]
    sr, size = prep["target_sr"], tuple(mc["image_size"])
    sos = pw.design_bandpass(prep["bandpass_low"], prep["bandpass_high"], sr, prep["butterworth_order"])
    backend = load_backend(a.model)

    combos = list(itertools.product([0, 1], repeat=3))       # (F, P, R)
    res = {c: [] for c in combos}
    names = []
    for it in golden["items"]:
        audio, fs = load_audio_mono(d / "wavs" / it["rn_filename"])
        segs = segment_audio(resample_to_target(audio, fs, sr), sr * prep["segment_length_s"])
        names.append(it["rn_filename"])
        for c in combos:
            res[c].append(float(np.mean(backend.predict_batch([image(s, sos, mc, sr, size, *c) for s in segs]))))
    py = np.array([it["python_mean_prob"] for it in golden["items"]])
    phone = np.array([rn[n]["rn_mean_prob"] for n in names])
    print(f"{'F P R':7s} {'|d| vs Python':>22s} {'|d| vs phone':>22s} {'class = Python':>15s} {'class = phone':>14s}")
    for c in combos:
        p = np.array(res[c])
        dp, dr = np.abs(p - py), np.abs(p - phone)
        print(f"{' '.join(map(str, c)):7s} med {np.median(dp):.4f} max {np.max(dp):.3f}   med {np.median(dr):.4f} "
              f"max {np.max(dr):.3f}   {np.sum((p >= .5) == (py >= .5)):2d}/26 {np.sum((p >= .5) == (phone >= .5)):10d}/26")
    full = np.array(res[(1, 1, 1)])
    print("\nper file (Python | F | P | R | all three | phone):")
    for i, n in enumerate(names):
        print(f"  {n[:44]:44s} {py[i]:.3f} | {res[(1,0,0)][i]:.3f} | {res[(0,1,0)][i]:.3f} | {res[(0,0,1)][i]:.3f} | "
              f"{full[i]:.3f} | {phone[i]:.3f}")


if __name__ == "__main__":
    main()
