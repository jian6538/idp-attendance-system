"""
main_blink.py  —  Option 2: Active liveness via MediaPipe blink detection
=========================================================================
Student must blink ONCE naturally within 6 seconds to be considered live.
A printed photo / phone screen can NEVER blink → always blocked.

Setup:
    pip install mediapipe
    python main_blink.py
"""

from __future__ import annotations

import os, sys, time
from datetime import datetime
from typing import Dict, Optional

import cv2

from anti_spoof_blink import BlinkLiveness, _MP_AVAILABLE
from attendance_logger import get_today_count, log_attendance
from face_utils import (
    DEFAULT_MATCH_THRESHOLD, detect_faces, get_embedding,
    identify_face, load_all_embeddings, load_facenet, load_mtcnn,
)
from recognition_state import RecognitionState
from ui_overlay import (
    draw_confirmed_box, draw_cooldown_box, draw_hud,
    draw_scanning_box, draw_unknown_box, draw_welcome_card,
    _put_text_with_bg, COLOR_ORANGE, COLOR_WHITE,
)

HERE        = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(HERE, "database", "students.json")
EMB_DIR     = os.path.join(HERE, "embeddings")
WINDOW_NAME = "Attendance — Blink Liveness"
CAMERA_IDX  = 0
FRAME_W, FRAME_H   = 640, 480
WELCOME_CARD_SECS  = 3.0
CONFIRM_DURATION   = 2.5
COOLDOWN_MINUTES   = 60
MATCH_THRESHOLD    = DEFAULT_MATCH_THRESHOLD


def _largest(detections):
    return max(detections, key=lambda d: (d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1])) if detections else None


def _draw_blink_box(frame, box, status_text: str):
    """Orange box with blink prompt while waiting for liveness confirmation."""
    import cv2
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
    _put_text_with_bg(frame, status_text, (x1, max(18, y1 - 8)),
                      color=COLOR_ORANGE, scale=0.6, thickness=2)


def main() -> int:
    print("=== Blink Detection Liveness Mode ===")
    if not _MP_AVAILABLE:
        print("ERROR: mediapipe not installed.  Run: pip install mediapipe"); return 1

    print("Loading MTCNN + FaceNet...")
    try:
        mtcnn, facenet = load_mtcnn(), load_facenet()
    except Exception as e:
        print(f"ERROR: {e}"); return 1

    blink_detector = BlinkLiveness()
    known = load_all_embeddings(DB_PATH, EMB_DIR)
    print(f"Loaded {len(known)} student(s).")
    state = RecognitionState(CONFIRM_DURATION, COOLDOWN_MINUTES)

    cap = cv2.VideoCapture(CAMERA_IDX)
    if not cap.isOpened():
        print("ERROR: cannot open camera."); return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    welcome_data: Optional[Dict] = None
    welcome_t: float = 0.0

    # Track previous candidate to reset blink state on change
    prev_candidate: Optional[str] = None
    print("Running. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None: continue
            frame = cv2.flip(frame, 1)

            detections = detect_faces(frame, mtcnn)
            target = _largest(detections)

            if target is None:
                # Reset blink state if no face
                if prev_candidate:
                    blink_detector.reset(prev_candidate)
                    prev_candidate = None
                state.update(None)
            else:
                box = target["box"]

                # ── Recognition first (need ID to track blink state per person) ─
                try:
                    emb = get_embedding(target["face_tensor"], facenet)
                except Exception:
                    emb = None

                match = identify_face(emb, known, MATCH_THRESHOLD) if emb is not None else None
                candidate_id = match["matrix_number"] if match else "unknown"

                # Reset blink state if candidate changed
                if candidate_id != prev_candidate:
                    if prev_candidate:
                        blink_detector.reset(prev_candidate)
                    prev_candidate = candidate_id

                # ── Liveness gate ─────────────────────────────────────────
                live, blink_text = blink_detector.check(frame, candidate_id)

                if not live:
                    _draw_blink_box(frame, box, blink_text)
                    state.update(None)
                    draw_hud(frame, get_today_count(), "EE4001 — Blink Liveness")
                    cv2.imshow(WINDOW_NAME, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"): break
                    continue

                # ── Attendance logic ──────────────────────────────────────
                if match is None:
                    draw_unknown_box(frame, box); state.update(None)
                else:
                    name, mn = match["name"], match["matrix_number"]
                    if state.is_on_cooldown(mn):
                        draw_cooldown_box(frame, box, name); state.update(None)
                    else:
                        status = state.update(mn)
                        if status == "confirmed":
                            if log_attendance(name, mn):
                                welcome_data = {"name": name, "matrix_number": mn,
                                                "timestamp": datetime.now()}
                                welcome_t = time.time()
                            state.mark_done(mn)
                            blink_detector.reset(mn)   # clear so next person starts fresh
                            prev_candidate = None
                            draw_confirmed_box(frame, box, name, mn)
                        elif status == "confirming":
                            draw_scanning_box(frame, box, state.get_progress(), f"Scanning... {name}")
                        else:
                            draw_scanning_box(frame, box, 0.0, "Scanning...")

            if welcome_data:
                if time.time() - welcome_t < WELCOME_CARD_SECS:
                    draw_welcome_card(frame, welcome_data["name"],
                                      welcome_data["matrix_number"], welcome_data["timestamp"])
                else:
                    welcome_data = None

            draw_hud(frame, get_today_count(), "EE4001 — Blink Liveness")
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        cap.release(); cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
