#!/usr/bin/env python3
"""
eeg_analysis.py
───────────────
FFT-based brainwave band extraction from 4-channel EEG data.

Pipeline
  1. Load        — read F1–F4 from semicolon-delimited CSV
  2. Filter      — zero-phase 4th-order Butterworth bandpass (1–50 Hz)
  3. Segment     — split into chronological 15-minute windows
  4. FFT         — Hann-windowed FFT → one-sided PSD per channel
  5. Band power  — trapezoidal integration over δ/θ/α/β/γ bands
  6. Score       — Cognitive Load = β/α (normalised 0–100)
                   Calmness       = α dominance (normalised 0–100)
  7. Output      — JSON array, one object per 15-min slice

Sampling rate  : 256 Hz
Window         : 15 min  = 900 s  = 230,400 samples
"""

import sys
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Configuration ──────────────────────────────────────────────────────────
FS               = 256            # sampling rate (Hz)
WINDOW_SEC       = 15 * 60       # 900 s per window
SAMPLES_PER_WIN  = FS * WINDOW_SEC  # 230,400
CHANNELS         = ["F1", "F2", "F3", "F4"]
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 50.0),
}
RECORDING_START  = datetime(2024, 1, 15, 0, 0, 0)
CSV_PATH         = Path("eeg_generated_daily.csv")
OUTPUT_PATH      = Path("eeg_band_analysis.json")

DIVIDER = "─" * 60


# ══════════════════════════════════════════════════════════════════════════
# Step 1 — Load
# ══════════════════════════════════════════════════════════════════════════
def load_csv(path: Path) -> np.ndarray:
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {path}")

    df = pd.read_csv(path, sep=";")
    missing = [c for c in CHANNELS if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Missing columns in CSV: {missing}")

    data = df[CHANNELS].dropna().to_numpy(dtype=np.float64)
    print(f"  Loaded   {len(data):>10,} samples × {data.shape[1]} channels")
    print(f"  Duration {len(data) / FS:>10,.2f} s  ({len(data) / FS / 60:.2f} min)")
    return data


# ══════════════════════════════════════════════════════════════════════════
# Step 2 — Bandpass filter
# ══════════════════════════════════════════════════════════════════════════
def bandpass_filter(
    data: np.ndarray,
    low: float = 1.0,
    high: float = 50.0,
    fs: int = FS,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase 4th-order Butterworth bandpass applied per channel via
    scipy.signal.filtfilt (forward + backward pass → no phase distortion).

    padlen is clamped to len(data) - 1 so the filter works on short arrays
    (e.g. the 96-row epoch CSV) without raising a ValueError.
    """
    nyq = fs / 2.0
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")

    # scipy default padlen = 3 * max(len(a)-1, len(b)-1)
    default_padlen = 3 * (max(len(a), len(b)) - 1)
    safe_padlen    = min(default_padlen, len(data) - 1)

    if safe_padlen < default_padlen:
        print(
            f"  [WARN] Only {len(data)} samples available — padlen reduced to "
            f"{safe_padlen} (default {default_padlen}).\n"
            f"         Frequency resolution will be coarse. Supply full 256 Hz "
            f"data for accurate band analysis."
        )

    out = np.empty_like(data)
    for i in range(data.shape[1]):
        out[:, i] = signal.filtfilt(b, a, data[:, i], padlen=safe_padlen)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Step 3 — Segment into 15-min windows
# ══════════════════════════════════════════════════════════════════════════
def segment_windows(data: np.ndarray) -> list[np.ndarray]:
    n_samples  = len(data)
    n_complete = n_samples // SAMPLES_PER_WIN

    if n_complete == 0:
        print(
            f"  [WARN] {n_samples} samples = {n_samples / FS:.2f}s of data.\n"
            f"         A 15-min window needs {SAMPLES_PER_WIN:,} samples.\n"
            f"         Treating entire dataset as a single window (demo mode)."
        )
        return [data]

    windows   = [data[i * SAMPLES_PER_WIN:(i + 1) * SAMPLES_PER_WIN] for i in range(n_complete)]
    remainder = n_samples % SAMPLES_PER_WIN
    if remainder:
        print(f"  Dropped {remainder} trailing samples (incomplete final window)")

    print(f"  Segmented into {n_complete} × 15-min windows")
    return windows


# ══════════════════════════════════════════════════════════════════════════
# Step 4 & 5 — FFT → PSD → band power
# ══════════════════════════════════════════════════════════════════════════
def compute_band_powers(window: np.ndarray, fs: int = FS) -> dict[str, float]:
    """
    For each channel:
      - Multiply by a Hann window to suppress spectral leakage
      - Compute FFT via scipy.fft.fft
      - Build one-sided PSD: PSD[k] = (2 / (N·fs)) · |X[k]|²  (µV²/Hz)
      - Integrate PSD within each band using the trapezoidal rule

    Returns the mean power across all channels for every band.
    """
    n        = len(window)
    hann     = np.hanning(n)                  # Hann taper
    freqs    = fftfreq(n, d=1.0 / fs)         # full (two-sided) frequency axis
    pos_mask = freqs > 0                       # keep positive frequencies only
    freqs_p  = freqs[pos_mask]

    band_powers: dict[str, float] = {}

    for band_name, (f_lo, f_hi) in BANDS.items():
        band_mask = (freqs_p >= f_lo) & (freqs_p <= f_hi)
        channel_powers: list[float] = []

        for ch in range(window.shape[1]):
            # Apply Hann taper and compute FFT
            spectrum = fft(window[:, ch] * hann)

            # One-sided amplitude-corrected PSD (µV²/Hz)
            # Factor of 2 compensates for discarding the negative-frequency half
            psd = (2.0 / (n * fs)) * np.abs(spectrum[pos_mask]) ** 2

            # Band power via trapezoidal integration
            n_bins = int(band_mask.sum())
            if n_bins == 0:
                power = 0.0
            elif n_bins == 1:
                # Only one FFT bin in this band — estimate area as value × bin width
                power = float(psd[band_mask][0] * (fs / n))
            else:
                power = float(trapezoid(psd[band_mask], freqs_p[band_mask]))
            channel_powers.append(power)

        band_powers[band_name] = float(np.mean(channel_powers))

    return band_powers


# ══════════════════════════════════════════════════════════════════════════
# Step 6 — Normalise to 0-100
# ══════════════════════════════════════════════════════════════════════════
def minmax_scale(
    values: list[float],
    invert: bool = False,
    fallback_range: tuple[float, float] = (0.0, 1.0),
) -> list[float]:
    """
    Min-max normalise values to [0, 100].

    When all values are identical (e.g. only one window), the observed range
    collapses to zero and we fall back to `fallback_range` so the score is
    still meaningful rather than a flat 0.
    """
    arr    = np.array(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    span   = hi - lo

    if span < 1e-12:
        ref_lo, ref_hi = fallback_range
        ref_span = (ref_hi - ref_lo) if (ref_hi - ref_lo) > 1e-12 else 1e-12
        scaled = np.clip((arr - ref_lo) / ref_span * 100.0, 0.0, 100.0)
    else:
        scaled = (arr - lo) / span * 100.0

    if invert:
        scaled = 100.0 - scaled

    return [round(float(v), 1) for v in scaled]


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(f"\n{DIVIDER}")
    print("  Mindflow EEG Band Analyser")
    print(f"{DIVIDER}")

    # ── 1. Load ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    raw = load_csv(CSV_PATH)

    # ── 2. Bandpass filter ───────────────────────────────────────────────
    print("\n[2/5] Bandpass filtering (1–50 Hz, 4th-order Butterworth, zero-phase)...")
    filtered = bandpass_filter(raw, low=1.0, high=50.0, fs=FS, order=4)
    print("  Filter applied to all channels")

    # ── 3. Segment ───────────────────────────────────────────────────────
    print("\n[3/5] Segmenting into 15-minute chronological windows...")
    windows = segment_windows(filtered)

    # ── 4 & 5. FFT + band powers ─────────────────────────────────────────
    print(f"\n[4/5] Running FFT on {len(windows)} window(s)...")
    print(f"      Frequency resolution: {FS / max(len(w) for w in windows):.4f} Hz/bin")

    raw_records: list[dict] = []

    for i, win in enumerate(windows):
        t_start  = RECORDING_START + timedelta(minutes=15 * i)
        t_end    = t_start + timedelta(minutes=15)
        powers   = compute_band_powers(win, fs=FS)

        alpha    = powers["alpha"]
        beta     = powers["beta"]
        total    = sum(powers.values()) + 1e-12

        # Raw metrics (normalised in next step)
        beta_alpha_ratio = beta / (alpha + 1e-12)  # cognitive load proxy
        alpha_dominance  = alpha / total             # calmness proxy

        raw_records.append({
            "window":            i + 1,
            "timeSlot":          t_start.strftime("%H:%M"),
            "startTime":         t_start.isoformat(),
            "endTime":           t_end.isoformat(),
            "bands":             {k: round(v, 8) for k, v in powers.items()},
            "_beta_alpha_ratio": beta_alpha_ratio,
            "_alpha_dominance":  alpha_dominance,
        })

        pct = (i + 1) / len(windows) * 100
        print(f"      [{pct:5.1f}%] Window {i+1:>3}/{len(windows)}  "
              f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}  "
              f"β/α={beta_alpha_ratio:.3f}  α_dom={alpha_dominance:.3f}",
              end="\r")

    print()  # newline after progress line

    # ── 6. Normalise ─────────────────────────────────────────────────────
    print("\n[5/5] Normalising Cognitive Load and Calmness to 0–100...")

    cog_raw   = [r["_beta_alpha_ratio"] for r in raw_records]
    calm_raw  = [r["_alpha_dominance"]  for r in raw_records]

    # Cognitive Load : high β/α ratio      → high score
    #   empirical fallback range: 0.2 (deep sleep) – 5.0 (peak focus)
    # Calmness       : high α dominance    → high score
    #   empirical fallback range: 0.05 (low alpha) – 0.60 (dominant alpha)
    cog_scores  = minmax_scale(cog_raw,  invert=False, fallback_range=(0.2,  5.0))
    calm_scores = minmax_scale(calm_raw, invert=False, fallback_range=(0.05, 0.60))

    # ── 7. Build final output ─────────────────────────────────────────────
    output: list[dict] = []
    for r, cog, calm in zip(raw_records, cog_scores, calm_scores):
        output.append({
            "window":    r["window"],
            "timeSlot":  r["timeSlot"],
            "startTime": r["startTime"],
            "endTime":   r["endTime"],
            "bands": {
                "delta_uV2_Hz": r["bands"]["delta"],
                "theta_uV2_Hz": r["bands"]["theta"],
                "alpha_uV2_Hz": r["bands"]["alpha"],
                "beta_uV2_Hz":  r["bands"]["beta"],
                "gamma_uV2_Hz": r["bands"]["gamma"],
            },
            "cognitiveLoad": cog,   # β/α ratio, min-max scaled 0–100
            "calmness":      calm,  # α/(total power), min-max scaled 0–100
        })

    # ── Save ──────────────────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"  Done. {len(output)} window(s) written → '{OUTPUT_PATH}'")
    print(DIVIDER)

    if output:
        peak_cog  = max(output, key=lambda r: r["cognitiveLoad"])
        peak_calm = max(output, key=lambda r: r["calmness"])
        print(f"\n  Peak Cognitive Load : {peak_cog['cognitiveLoad']:5.1f}  @ {peak_cog['timeSlot']}")
        print(f"  Peak Calmness       : {peak_calm['calmness']:5.1f}  @ {peak_calm['timeSlot']}")

    print(f"\n  Sample output (first window):\n")
    print(json.dumps(output[0] if output else {}, indent=4))
    print()


if __name__ == "__main__":
    main()
