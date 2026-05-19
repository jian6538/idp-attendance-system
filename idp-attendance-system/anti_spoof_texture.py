"""
anti_spoof_texture.py
=====================
Passive liveness via classical texture analysis — NO model file needed.

Uses two complementary signals on the face crop:

1. Laplacian Variance (blur/sharpness)
   Real faces captured by a webcam have a natural sharpness range.
   A phone screen held up produces a Moiré pattern; printed photos are
   often slightly blurred and have very uniform texture.

2. FFT High-Frequency Ratio
   Printed attacks and screen replays introduce periodic patterns
   (printer dots, pixel grids) that show up as spikes in the Fourier
   spectrum at mid-to-high frequencies.  Real faces have a smoother,
   more natural frequency distribution.

Both signals are combined into a single liveness score [0, 1].
Tuning is done via the thresholds below — run with debug=True in
is_live_texture() to print values while calibrating.

No extra pip packages needed — only OpenCV + numpy (already installed).
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Tunable thresholds — adjust these if the detector is over/under-triggering
# ---------------------------------------------------------------------------
LAP_MIN   = 30.0    # Laplacian variance below this → too blurry / printed flat
LAP_MAX   = 2500.0  # above this → screen Moiré noise
FFT_RATIO_MAX = 0.35  # fraction of energy in high frequencies; screens/prints > 0.35
SCORE_THRESHOLD = 0.45  # combined score must exceed this to be considered live


def _laplacian_variance(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _fft_high_freq_ratio(gray: np.ndarray) -> float:
    """Fraction of FFT energy in the outer 50% of the frequency plane."""
    f    = np.fft.fft2(gray.astype(np.float32))
    fsh  = np.fft.fftshift(f)
    mag  = np.abs(fsh) ** 2

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    # Inner circle radius = 25% of min dimension → low-frequency energy
    r    = min(h, w) // 4
    Y, X = np.ogrid[:h, :w]
    mask = (Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2

    total      = mag.sum() + 1e-8
    low_energy = mag[mask].sum()
    high_ratio = 1.0 - (low_energy / total)
    return float(high_ratio)


def _crop_face(frame_bgr: np.ndarray, box: List[int],
               scale: float = 1.5, size: int = 64) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw = (x2 - x1) * scale; bh = (y2 - y1) * scale
    nx1 = max(0, int(cx - bw / 2)); ny1 = max(0, int(cy - bh / 2))
    nx2 = min(w, int(cx + bw / 2)); ny2 = min(h, int(cy + bh / 2))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    crop = frame_bgr[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size))


def is_live_texture(
    frame_bgr: np.ndarray,
    box: List[int],
    debug: bool = False,
) -> Tuple[bool, float]:
    """
    Returns (is_live: bool, score: float 0–1).

    score close to 1 → likely real face
    score close to 0 → likely spoof (print / screen)
    """
    gray = _crop_face(frame_bgr, box)
    if gray is None:
        return True, 1.0

    lap   = _laplacian_variance(gray)
    fft_r = _fft_high_freq_ratio(gray)

    # Lap score: 1.0 when lap is in the natural face range, drops toward 0 outside
    if lap < LAP_MIN:
        lap_score = lap / LAP_MIN                          # too blurry
    elif lap > LAP_MAX:
        lap_score = LAP_MAX / lap                          # screen Moiré
    else:
        lap_score = 1.0

    # FFT score: 1.0 when high-freq ratio is low (natural face), 0 for screen/print
    fft_score = max(0.0, 1.0 - (fft_r / FFT_RATIO_MAX))

    # Weighted combination
    score = 0.5 * lap_score + 0.5 * fft_score
    score = float(np.clip(score, 0.0, 1.0))

    if debug:
        print(f"[Texture] lap={lap:.1f}  fft_ratio={fft_r:.3f}  "
              f"lap_score={lap_score:.2f}  fft_score={fft_score:.2f}  "
              f"combined={score:.2f}")

    return score >= SCORE_THRESHOLD, score
