"""
anti_spoof.py
=============
Face liveness detection using the hairymax/Face-AntiSpoofing ONNX model.

GitHub : https://github.com/hairymax/Face-AntiSpoofing
Model  : AntiSpoofing_bin_1.5_128.onnx
         Binary classification — index 0 = Real, index 1 = Fake
Input  : (1, 3, 128, 128) float32 normalised to [0, 1]
Crop   : 1.5× enlarged bounding box with black border padding
Accuracy: 92.92% / AUC-ROC 0.987 (binary model)

Public API
----------
    load_anti_spoof_model(weights_path)  → ort.InferenceSession | None
    check_liveness(frame, box, session, threshold)  → (bool, float)

    check_liveness returns:
        (True,  real_prob)  — live person detected
        (False, real_prob)  — spoof / photo detected
        (True,  1.0)        — liveness disabled (session is None)
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

WEIGHTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anti_spoof_weights")
DEFAULT_WEIGHTS = os.path.join(WEIGHTS_DIR, "AntiSpoofing_bin_1.5_128.onnx")

INPUT_SIZE  = (128, 128)   # model expects 128×128
BBOX_INC    = 1.5          # enlarge face crop by 1.5×

# Real-face probability must be >= this to pass the liveness gate.
# 0.5 is the recommended threshold for this binary model.
DEFAULT_LIVENESS_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_anti_spoof_model(weights_path: str = DEFAULT_WEIGHTS):
    """Load AntiSpoofing_bin_1.5_128.onnx and return an ONNX InferenceSession.

    Returns None if onnxruntime is missing or the weights file does not exist.
    The main loop falls back to "no liveness check" rather than crashing.
    """
    if not _ORT_AVAILABLE:
        print(
            "[AntiSpoof] WARNING: 'onnxruntime' not installed.\n"
            "  Run:  pip install onnxruntime\n"
            "  Liveness check will be DISABLED."
        )
        return None

    if not os.path.isfile(weights_path):
        print(
            f"[AntiSpoof] WARNING: weights not found at:\n  {weights_path}\n"
            "  Run  python download_weights.py  to fetch them.\n"
            "  Liveness check will be DISABLED until weights are available."
        )
        return None

    try:
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 2
        sess_options.intra_op_num_threads = 2
        session = ort.InferenceSession(
            weights_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        # Smoke-test with a dummy input to catch shape/version issues early.
        dummy      = np.zeros((1, 3, 128, 128), dtype=np.float32)
        input_name = session.get_inputs()[0].name
        session.run(None, {input_name: dummy})
        print("[AntiSpoof] hairymax AntiSpoofing_bin_1.5_128.onnx loaded OK.")
        return session
    except Exception as e:
        print(f"[AntiSpoof] WARNING: failed to load model ({e}). Liveness DISABLED.")
        return None


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _increased_crop(
    img_bgr: np.ndarray,
    box: List[int],
    bbox_inc: float = BBOX_INC,
) -> Optional[np.ndarray]:
    """Enlarge the face bounding box by bbox_inc and crop with black padding.

    box  : [x1, y1, x2, y2]  (MTCNN format)
    Returns a BGR crop of the enlarged region, or None on failure.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    # Expand outward from bbox centre
    expand_w = int(bw * (bbox_inc - 1) / 2)
    expand_h = int(bh * (bbox_inc - 1) / 2)

    nx1 = x1 - expand_w
    ny1 = y1 - expand_h
    nx2 = x2 + expand_w
    ny2 = y2 + expand_h

    # Padding amounts for out-of-frame regions
    pad_left   = max(0, -nx1)
    pad_top    = max(0, -ny1)
    pad_right  = max(0, nx2 - w)
    pad_bottom = max(0, ny2 - h)

    # Clamp to image
    cx1, cy1 = max(0, nx1), max(0, ny1)
    cx2, cy2 = min(w, nx2), min(h, ny2)

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    # Add black border for out-of-frame portions
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0],
        )

    return crop


def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR crop to (1, 3, 128, 128) float32 tensor in [0, 1]."""
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)          # BGR → RGB
    resized = cv2.resize(rgb, INPUT_SIZE)                         # 128×128
    tensor  = resized.astype(np.float32) / 255.0                  # [0, 1]
    tensor  = tensor.transpose(2, 0, 1)[np.newaxis, ...]          # (1,3,128,128)
    return np.ascontiguousarray(tensor)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def check_liveness(
    frame_bgr: np.ndarray,
    box: List[int],
    session,
    threshold: float = DEFAULT_LIVENESS_THRESHOLD,
) -> Tuple[bool, float]:
    """Run liveness check on one face.

    Parameters
    ----------
    frame_bgr : full camera frame (BGR numpy array)
    box       : [x1, y1, x2, y2] bounding box from MTCNN
    session   : ONNX InferenceSession (or None to disable)
    threshold : real-face probability threshold (default 0.5)

    Returns
    -------
    (is_live, real_prob)
        is_live   — True = real person, False = spoof/photo
        real_prob — confidence score for display/debugging
    """
    if session is None:
        return True, 1.0   # liveness disabled

    crop = _increased_crop(frame_bgr, box)
    if crop is None:
        return True, 1.0   # crop failed — don't block

    try:
        tensor     = _preprocess(crop)
        input_name = session.get_inputs()[0].name
        logits     = session.run(None, {input_name: tensor})[0]  # (1, 2)
        probs      = _softmax(logits)[0]                          # (2,)

        # Binary model: index 0 = Real, index 1 = Fake
        real_prob = float(probs[0])
        return real_prob >= threshold, real_prob

    except Exception:
        return True, 1.0   # fail open on inference error
