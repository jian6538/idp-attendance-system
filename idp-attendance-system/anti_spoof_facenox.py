"""
anti_spoof_facenox.py
=====================
Liveness detection using facenox/face-antispoof-onnx.

Model : MiniFASNetV2-SE (quantized INT8, ~600 KB)
Input : (1, 3, 128, 128) float32 normalised to [0, 1]
Output: 2-class softmax — index 0 = Real, index 1 = Spoof
Accuracy: 98.2% on 70k+ CelebA-Spoof samples

Download weights first:
    python download_weights_facenox.py
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

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anti_spoof_weights")
DEFAULT_WEIGHTS_FACENOX = os.path.join(WEIGHTS_DIR, "facenox_quantized.onnx")

CROP_SCALE   = 2.7
INPUT_SIZE   = (128, 128)       # facenox uses 128×128 (not 80×80)
LIVENESS_THR = 0.55             # real probability must exceed this


def load_facenox_model(weights_path: str = DEFAULT_WEIGHTS_FACENOX):
    if not _ORT_AVAILABLE:
        print("[Facenox] WARNING: onnxruntime not installed.  pip install onnxruntime")
        return None
    if not os.path.isfile(weights_path):
        print(f"[Facenox] WARNING: weights not found at {weights_path}\n"
              "  Run: python download_weights_facenox.py")
        return None
    try:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(weights_path, sess_options=opts,
                                       providers=["CPUExecutionProvider"])
        # smoke test
        dummy = np.zeros((1, 3, 128, 128), dtype=np.float32)
        session.run(None, {session.get_inputs()[0].name: dummy})
        return session
    except Exception as e:
        print(f"[Facenox] WARNING: failed to load model ({e})")
        return None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _crop(frame_bgr: np.ndarray, box: List[int]) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * CROP_SCALE, (y2 - y1) * CROP_SCALE
    nx1 = max(0, int(cx - bw / 2)); ny1 = max(0, int(cy - bh / 2))
    nx2 = min(w, int(cx + bw / 2)); ny2 = min(h, int(cy + bh / 2))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    crop = frame_bgr[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None
    rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rsz  = cv2.resize(rgb, INPUT_SIZE)
    return (rsz.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]


def is_live_facenox(frame_bgr, box, session, threshold=LIVENESS_THR) -> Tuple[bool, float]:
    if session is None:
        return True, 1.0
    tensor = _crop(frame_bgr, box)
    if tensor is None:
        return True, 1.0
    try:
        name   = session.get_inputs()[0].name
        logits = session.run(None, {name: tensor})[0]
        probs  = _softmax(logits)[0]
        # facenox: index 0 = Real, index 1 = Spoof
        real_prob = float(probs[0])
        return real_prob >= threshold, real_prob
    except Exception:
        return True, 1.0
