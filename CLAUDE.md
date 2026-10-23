# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Master's thesis pipeline for binary heart sound classification (Normal vs Abnormal) using MobileNetV2 transfer learning on Mel spectrograms. Trains on PhysioNet/CinC 2016 (training-a..e) plus PASCAL Dataset-B; ultimately deployed to Android via TFLite from a MEMS stethoscope app. The methodology in `claude_code_gelistirme_plani.md` is the source of truth for parameter choices — those choices come from the thesis text (Chapters 3.1 and 3.4) and must not be changed casually. Read that file before altering preprocessing constants, model topology, or training schedule.

## Pipeline stages and where they run

The project is a strictly ordered pipeline. Stages 0–6 run locally (Windows / PowerShell). Stages 7–10 run on **Google Colab with a T4 GPU** — TensorFlow training is not intended for the local machine.

```
download → resample_segment → clean_filter → mel_spectrogram → split  (local)
                                          → augmentation (on-the-fly during train)
                                          → build_model → train → evaluate → convert_tflite  (Colab)
```

Each stage reads its inputs from a CSV produced by the previous stage and writes the next CSV. The CSV chain is the contract between stages:

`data/metadata.csv` → `data/segments_metadata.csv` → `data/processed_metadata.csv` → `data/mel_metadata.csv` → `splits/{train,val,test}.csv`

Every row carries `patient_id` end-to-end. **Never break patient_id through the pipeline** — it is required for `GroupShuffleSplit` in stage 6 to prevent patient-level data leakage between train/val/test.

## Configuration

All parameters live in `configs/config.yaml` and are loaded with `load_config()` at the top of every module. Do not hardcode sample rates, segment lengths, filter cutoffs, Mel parameters, or training hyperparameters — change them in `config.yaml`. Class weights are computed by `src/data/split.py` and written to `configs/class_weights.json`; the training loop reads them from there.

Key invariants (see `config.yaml`):
- Target sample rate: 4000 Hz. PhysioNet (2000 Hz) is upsampled via `scipy.signal.resample_poly(up=2, down=1)`; PASCAL is left alone.
- Segments: 5 s = 20,000 samples. Tail shorter than 5 s is dropped.
- Bandpass: 4th-order Butterworth, 25–900 Hz, `sosfiltfilt` (zero-phase).
- Mel: `n_fft=512, hop_length=256, n_mels=64, fmin=25, fmax=900`, then stacked to 3 channels and resized to 224×224 for MobileNetV2.

## Running stages

Every module is invoked as a package from the project root and accepts standard flags:

```powershell
python -m src.data.download                   # full download + metadata
python -m src.data.download --dry-run         # show what would happen
python -m src.data.download --verify          # check expected file counts
python -m src.preprocessing.resample_segment
python -m src.preprocessing.clean_filter
python -m src.preprocessing.mel_spectrogram
python -m src.preprocessing.mel_spectrogram --figures-only   # only regenerate thesis figures
python -m src.data.split
```

Common flags across preprocessing modules: `--dry-run`, `--verify`, `--figures-only`. Use `--figures-only` to regenerate thesis figures in `figures/` without rerunning heavy processing.

Training, evaluation, and TFLite conversion expect Colab paths:

```bash
python -m src.training.train       --data-dir /content/data --output-dir /content/output
python -m src.evaluation.evaluate  --data-dir /content/data --model-path /content/output/models/best_model.keras --output-dir /content/output
python -m src.deployment.convert_tflite --model-path /content/output/models/best_model.keras --data-dir /content/data --output-dir /content/output
```

## Local → Colab handoff

`scripts/package_for_colab.py` mirrors `src/`, `configs/`, `splits/` and a zipped `data/mel/` into `drive_upload/`. That directory is intentionally a duplicate of source code — when modifying preprocessing or model code, regenerate `drive_upload/` rather than editing it in place (otherwise local and Colab copies diverge silently). The Colab notebook lives at `notebooks/train_colab.ipynb`.

## Dependencies

`requirements.txt` pins TensorFlow to 2.15.*. Local stages 0–6 do not need TensorFlow at all (only NumPy/SciPy/librosa/soundfile/scikit-learn); installing TF locally is optional and often skipped because training happens on Colab.

## Label fix (2026-10-20)

PhysioNet 2016 REFERENCE.csv uses 1 = abnormal, -1 = normal. Until 2026-10-20 `config.yaml` mapped them the other way round,
so every PhysioNet label in the CSV chain was inverted and the deployed model (thesis Table 4.x, 87.03 %) was trained on that.
`scripts/fix_physionet_labels.py` rewrites the labels from REFERENCE.csv (checked against every .hea header; idempotent) and
recomputes `class_weights.json`. `build_model.py` now has a Rescaling layer ([0,1] -> [-1,1]) in front of MobileNetV2, and
`unfreeze_for_stage2` finds the base by name. Until v2 the augmentation in `augmentation.py` was never used in training.
Since 2026-10-21 `train.py` (`make_augmenter`) applies the same four methods on the fly to the Mel images, scaled to
pixels, when `training.augmentation.enabled` is true (v3); the notebook passes `aug_cfg` to `create_dataset`.
Model versions: v1 = inverted labels (output_v1_ters_etiket on Drive), v2 = fixed labels, no augmentation
(`output/v2_label_fix_20261021/`), v3 = fixed labels + augmentation.

## Things that look like bugs but aren't

- `main.py` is a leftover PyCharm scaffold. The real entry points are the `python -m src...` modules. Don't treat `main.py` as a launcher.
- `drive_upload/src/...` files duplicate `src/...`. They exist for Colab packaging, not as a separate codebase.
- The repository is on branch `master` while PRs target `main` (per gitStatus).
