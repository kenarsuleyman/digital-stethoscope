"""
Data Augmentation Pipeline for Heart Sound Classification

Applied ON-THE-FLY to training set only. Four augmentation techniques:
1. Time shift: shift raw PCG ±1000 samples (zero-padded), recompute Mel
2. Time masking: zero out 5-15 consecutive time steps in Mel
3. Frequency masking: zero out 2-8 consecutive Mel bands
4. Random amplitude scaling: multiply Mel by U(0.8, 1.2)

Usage:
    python -m src.preprocessing.augmentation --figures-only   # Generate thesis figure
"""

import argparse
import csv
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Augmentation functions (pure NumPy — can be wrapped in tf.py_function)
# ---------------------------------------------------------------------------

def time_shift(audio: np.ndarray, max_shift: int = 1000) -> np.ndarray:
    """Shift audio by a random amount in [-max_shift, +max_shift], zero-pad edges."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    shifted = np.zeros_like(audio)
    if shift > 0:
        shifted[shift:] = audio[:-shift]
    elif shift < 0:
        shifted[:shift] = audio[-shift:]
    else:
        shifted = audio.copy()
    return shifted


def time_mask(mel: np.ndarray, min_steps: int = 5, max_steps: int = 15) -> np.ndarray:
    """Zero out a random consecutive block of time steps in Mel spectrogram."""
    mel = mel.copy()
    n_time = mel.shape[1]
    mask_len = np.random.randint(min_steps, max_steps + 1)
    start = np.random.randint(0, max(1, n_time - mask_len))
    mel[:, start:start + mask_len] = 0.0
    return mel


def freq_mask(mel: np.ndarray, min_bands: int = 2, max_bands: int = 8) -> np.ndarray:
    """Zero out a random consecutive block of frequency bands in Mel spectrogram."""
    mel = mel.copy()
    n_mels = mel.shape[0]
    mask_len = np.random.randint(min_bands, max_bands + 1)
    start = np.random.randint(0, max(1, n_mels - mask_len))
    mel[start:start + mask_len, :] = 0.0
    return mel


def amplitude_scale(mel: np.ndarray, low: float = 0.8, high: float = 1.2) -> np.ndarray:
    """Multiply entire Mel spectrogram by a random factor from U(low, high)."""
    factor = np.random.uniform(low, high)
    return mel * factor


def augment_audio_to_mel(audio: np.ndarray, sr: int, mel_config: dict,
                         image_size: tuple) -> np.ndarray:
    """
    Apply a random augmentation pipeline to audio → Mel spectrogram image.

    Steps:
    1. Randomly apply time shift to raw audio
    2. Compute Mel spectrogram
    3. Convert to dB, normalize [0,1]
    4. Randomly apply one of: time mask, freq mask, amplitude scale
    5. Resize and stack to 3-channel

    Returns: (H, W, 3) float32 array
    """
    from PIL import Image

    # 1. Time shift (50% chance)
    if np.random.random() < 0.5:
        audio = time_shift(audio)

    # 2. Mel spectrogram
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=mel_config["n_fft"],
        hop_length=mel_config["hop_length"],
        n_mels=mel_config["n_mels"],
        fmin=mel_config["fmin"],
        fmax=mel_config["fmax"],
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalize [0, 1]
    s_min, s_max = S_db.min(), S_db.max()
    if s_max - s_min > 0:
        S_norm = (S_db - s_min) / (s_max - s_min)
    else:
        S_norm = np.zeros_like(S_db)

    # 3. Randomly apply one spectral augmentation
    aug_choice = np.random.randint(0, 3)
    if aug_choice == 0:
        S_norm = time_mask(S_norm)
    elif aug_choice == 1:
        S_norm = freq_mask(S_norm)
    else:
        S_norm = amplitude_scale(S_norm)

    S_norm = np.clip(S_norm, 0.0, 1.0).astype(np.float32)

    # 4. Resize and stack
    img = Image.fromarray(S_norm)
    img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    S_resized = np.array(img, dtype=np.float32)
    return np.stack([S_resized, S_resized, S_resized], axis=-1)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figures(config: dict, project_root: Path):
    """Generate fig_augmentation_examples.png for thesis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", palette="colorblind", font_scale=1.0)
    plt.rcParams["font.family"] = "DejaVu Sans"

    mel_config = config["mel_spectrogram"]
    target_sr = config["preprocessing"]["target_sr"]
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Pick a processed segment
    processed_meta_path = project_root / config["paths"]["processed_metadata_csv"]
    with open(processed_meta_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Use an abnormal segment for more visual interest
    example_row = next(r for r in rows if r["binary_label"] == "1")
    audio, sr = sf.read(project_root / example_row["segment_path"], dtype="float64")

    # Compute base Mel spectrogram (dB, normalized [0,1])
    def compute_mel_norm(audio_in):
        S = librosa.feature.melspectrogram(
            y=audio_in, sr=sr,
            n_fft=mel_config["n_fft"],
            hop_length=mel_config["hop_length"],
            n_mels=mel_config["n_mels"],
            fmin=mel_config["fmin"],
            fmax=mel_config["fmax"],
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        s_min, s_max = S_db.min(), S_db.max()
        if s_max - s_min > 0:
            return (S_db - s_min) / (s_max - s_min)
        return np.zeros_like(S_db)

    # Fix seed for reproducible figure
    np.random.seed(42)

    # (1) Original
    mel_orig = compute_mel_norm(audio)

    # (2) Time-shifted
    audio_shifted = time_shift(audio, max_shift=1000)
    mel_shifted = compute_mel_norm(audio_shifted)

    # (3) Time-masked
    mel_tmask = time_mask(mel_orig, min_steps=8, max_steps=12)

    # (4) Frequency-masked
    mel_fmask = freq_mask(mel_orig, min_bands=4, max_bands=6)

    # (5) Amplitude-scaled
    np.random.seed(99)
    mel_amp = amplitude_scale(mel_orig, low=0.8, high=1.2)
    mel_amp = np.clip(mel_amp, 0.0, 1.0)

    # Plot
    titles = [
        "(1) Orijinal",
        "(2) Zaman Kaydırma",
        "(3) Zaman Maskeleme",
        "(4) Frekans Maskeleme",
        "(5) Genlik Ölçekleme",
    ]
    mels = [mel_orig, mel_shifted, mel_tmask, mel_fmask, mel_amp]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    hop = mel_config["hop_length"]

    for ax, mel, title in zip(axes, mels, titles):
        librosa.display.specshow(
            mel, sr=target_sr, hop_length=hop,
            x_axis="time", y_axis="mel",
            fmin=mel_config["fmin"], fmax=mel_config["fmax"],
            ax=ax, cmap="magma", vmin=0, vmax=1,
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Zaman (s)", fontsize=9)
        if ax != axes[0]:
            ax.set_ylabel("")

    axes[0].set_ylabel("Frekans (Hz)", fontsize=10)
    fig.suptitle("Veri Artırma Teknikleri — Mel Spektrogram Örnekleri",
                 fontsize=14, fontweight="bold", y=1.04)
    plt.tight_layout()
    path = figures_dir / "fig_augmentation_examples.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def generate_pairwise_figures(config: dict, project_root: Path):
    """Generate four figures, each pairing the original Mel spectrogram with one
    augmentation method side by side (original | augmented)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", palette="colorblind", font_scale=1.0)
    plt.rcParams["font.family"] = "DejaVu Sans"

    mel_config = config["mel_spectrogram"]
    target_sr = config["preprocessing"]["target_sr"]
    hop = mel_config["hop_length"]
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Pick one processed segment (abnormal for more visual interest)
    processed_meta_path = project_root / config["paths"]["processed_metadata_csv"]
    with open(processed_meta_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    example_row = next(r for r in rows if r["binary_label"] == "1")
    audio, sr = sf.read(project_root / example_row["segment_path"], dtype="float64")

    def compute_mel_norm(audio_in):
        S = librosa.feature.melspectrogram(
            y=audio_in, sr=sr,
            n_fft=mel_config["n_fft"],
            hop_length=mel_config["hop_length"],
            n_mels=mel_config["n_mels"],
            fmin=mel_config["fmin"],
            fmax=mel_config["fmax"],
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        s_min, s_max = S_db.min(), S_db.max()
        if s_max - s_min > 0:
            return (S_db - s_min) / (s_max - s_min)
        return np.zeros_like(S_db)

    # Original (shared across all four figures)
    np.random.seed(42)
    mel_orig = compute_mel_norm(audio)

    # Build the augmented Mel for each method.
    # For the figure we use deterministic values at (or near) the edge of each
    # augmentation's real range so the effect is clearly visible — a random draw
    # near the middle of the range (e.g. a 0.03 s shift or a 1.05x scale) is
    # technically correct but visually indistinguishable from the original.
    audio_shifted = np.zeros_like(audio)        # fixed +1000-sample (0.25 s) shift
    audio_shifted[1000:] = audio[:-1000]
    mel_shifted = compute_mel_norm(audio_shifted)
    mel_tmask = time_mask(mel_orig, min_steps=8, max_steps=12)
    mel_fmask = freq_mask(mel_orig, min_bands=4, max_bands=6)
    mel_amp = np.clip(mel_orig * 0.8, 0.0, 1.0)  # low extreme of U(0.8, 1.2)

    # (filename, augmented mel, augmented-panel title)
    specs = [
        ("fig_aug_time_shift.png",      mel_shifted, "Zaman Kaydırma"),
        ("fig_aug_time_mask.png",       mel_tmask,   "Zaman Maskeleme"),
        ("fig_aug_freq_mask.png",       mel_fmask,   "Frekans Maskeleme"),
        ("fig_aug_amplitude_scale.png", mel_amp,     "Genlik Ölçekleme"),
    ]

    for fname, mel_aug, aug_title in specs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, mel, title in zip(axes, [mel_orig, mel_aug],
                                  ["Orijinal", aug_title]):
            librosa.display.specshow(
                mel, sr=target_sr, hop_length=hop,
                x_axis="time", y_axis="mel",
                fmin=mel_config["fmin"], fmax=mel_config["fmax"],
                ax=ax, cmap="magma", vmin=0, vmax=1,
            )
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Zaman (s)", fontsize=10)
            if ax != axes[0]:
                ax.set_ylabel("")
        axes[0].set_ylabel("Frekans (Hz)", fontsize=10)
        fig.suptitle(f"Veri Artırma — {aug_title}",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        path = figures_dir / fname
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Augmentation pipeline & figure generation")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--figures-only", action="store_true",
                        help="Generate combined thesis figure")
    parser.add_argument("--pairwise-figures", action="store_true",
                        help="Generate four side-by-side original/augmented figures")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)

    if args.pairwise_figures:
        generate_pairwise_figures(config, PROJECT_ROOT)
        return

    if args.figures_only:
        generate_figures(config, PROJECT_ROOT)
        return

    # Default: just generate the figure (augmentation is applied on-the-fly during training)
    logger.info("Augmentation is applied on-the-fly during training (Stage 8).")
    logger.info("Use --figures-only to generate the thesis figure.")
    logger.info("Functions available for import: time_shift, time_mask, freq_mask, "
                "amplitude_scale, augment_audio_to_mel")
    generate_figures(config, PROJECT_ROOT)


if __name__ == "__main__":
    main()
