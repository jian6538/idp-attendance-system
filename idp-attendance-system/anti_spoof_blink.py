"""
anti_spoof_blink.py
===================
Active liveness detection via Eye Aspect Ratio (EAR) blink detection.

Uses MediaPipe FaceMesh (468 3-D landmarks) to track eye openness every
frame.  A student is considered LIVE once they naturally blink at least
BLINKS_REQUIRED times within WINDOW_SECONDS of first being detected.

No pretrained anti-spoof model file needed — MediaPipe is the only dep.

Install:
    pip install mediapipe

How it works:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    EAR drops below EAR_THRESHOLD during a blink → blink counted.
    Real people blink naturally; printed photos / screens never blink.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# MediaPipe FaceMesh landmark indices for each eye
# (p1, p2, p3, p4, p5, p6) — horizontal endpoints + vertical pairs
LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD  = 0.22    # below this → eye closed
BLINKS_REQUIRED = 1      # 1 natural blink to confirm liveness
WINDOW_SECONDS  = 6.0    # must blink within this window of first detection


class BlinkLiveness:
    """Per-track blink state. One instance shared across all detections."""

    def __init__(self) -> None:
        self._face_mesh = None
        # per-candidate tracking keyed by a string label (matrix_no or "unknown")
        self._first_seen: Dict[str, float] = {}
        self._blink_count: Dict[str, int]  = {}
        self._eye_was_closed: Dict[str, bool] = {}
        self._confirmed: Dict[str, bool]   = {}

    def _init_mesh(self):
        if not _MP_AVAILABLE:
            return None
        if self._face_mesh is None:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return self._face_mesh

    @staticmethod
    def _ear(landmarks, indices, w: int, h: int) -> float:
        pts = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in indices],
            dtype=np.float32,
        )
        # Vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        # Horizontal distance
        hz = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * hz + 1e-6)

    def check(
        self,
        frame_bgr: np.ndarray,
        candidate_id: str,
    ) -> Tuple[bool, str]:
        """
        Returns (is_live, status_text).
            is_live     — True once blinks confirmed
            status_text — short string for UI ("Blink! 0/1", "✓ Live", etc.)
        """
        mesh = self._init_mesh()
        if mesh is None:
            return True, "Liveness: N/A (install mediapipe)"

        now = time.time()

        # Reset tracking if candidate changes (handled externally via reset())
        if candidate_id not in self._first_seen:
            self._first_seen[candidate_id]    = now
            self._blink_count[candidate_id]   = 0
            self._eye_was_closed[candidate_id] = False
            self._confirmed[candidate_id]     = False

        # Already confirmed in this window
        if self._confirmed[candidate_id]:
            return True, f"Live ({self._blink_count[candidate_id]} blink(s))"

        # Window expired without enough blinks → reset
        elapsed = now - self._first_seen[candidate_id]
        if elapsed > WINDOW_SECONDS:
            self._first_seen[candidate_id]    = now
            self._blink_count[candidate_id]   = 0
            self._eye_was_closed[candidate_id] = False

        # Run FaceMesh on this frame
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_ear  = self._ear(lm, LEFT_EYE,  w, h)
            right_ear = self._ear(lm, RIGHT_EYE, w, h)
            avg_ear   = (left_ear + right_ear) / 2.0

            eye_closed = avg_ear < EAR_THRESHOLD

            if eye_closed and not self._eye_was_closed[candidate_id]:
                # Rising edge of a blink
                self._blink_count[candidate_id] += 1

            self._eye_was_closed[candidate_id] = eye_closed

        blinks = self._blink_count[candidate_id]
        if blinks >= BLINKS_REQUIRED:
            self._confirmed[candidate_id] = True
            return True, f"Live ({blinks} blink(s) detected)"

        remaining = max(0.0, WINDOW_SECONDS - elapsed)
        return False, f"Blink please!  {blinks}/{BLINKS_REQUIRED}  ({remaining:.1f}s)"

    def reset(self, candidate_id: str) -> None:
        """Call when a candidate disappears or changes so the window restarts."""
        for d in (self._first_seen, self._blink_count,
                  self._eye_was_closed, self._confirmed):
            d.pop(candidate_id, None)

    def reset_all(self) -> None:
        for d in (self._first_seen, self._blink_count,
                  self._eye_was_closed, self._confirmed):
            d.clear()
