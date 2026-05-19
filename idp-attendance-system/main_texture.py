"""
main_texture.py  —  Option 3: Passive liveness via texture analysis
====================================================================
No model file, no extra packages — uses only OpenCV + numpy.

Two signals per frame:
  • Laplacian variance  — detects blur / Moiré from screens
  • FFT high-freq ratio — detects periodic patterns from print/screen attacks

Scores are combined into a single 0–1 liveness value; threshold = 0.45.
Run with debug output by setting DEBUG_TEXTURE = True below.

Setup:
    python main_texture.py   (no downloads needed)
"""

from __future__ import annotations

import os, sys, time
from datetime import datetime
from typing import Dict, Optional

import cv2

from anti_spoof_texture import is_live_texture, SCORE_THRESHOLD
from attendance_logger import get_today_count, log_attendance
from face_utils import (
    DEFAULT_MATCH_THRESHOLD, detect_faces, get_embedding,
    identify_face, load_all_embeddings, load_facenet, load_mtcnn,
)
from recognition_state import RecognitionState
from ui_overlay import (
    draw_confirmed_box, draw_cooldown_box, draw_hud,
    draw_scanning_box, draw_spoof_box, draw_unknown_box, draw_welcome_card,
)

DEBUG_TEXTURE = False   # set True to print lap/fft values to console for tuning

HERE        = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(HERE, "database", "students.json")
EMB_DIR     = os.path.join(HERE, "embeddings")
WINDOW_NAME = "Attendance — Texture Liveness"
CAMERA_IDX  = 0
FRAME_W, FRAME_H   = 640, 480
WELCOME_CARD_SECS  = 3.0
CONFIRM_DURATION   = 2.5
COOLDOWN_MINUTES   = 60
MATCH_THRESHOLD    = DEFAULT_MATCH_THRESHOLD


def _largest(detections):
    return max(detections, key=lambda d: (d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1])) if detections else None


def main() -> int:
    print("=== Texture Analysis Liveness Mode ===")
    print("No model needed — using Laplacian + FFT texture analysis.")
    print(f"Liveness threshold: {SCORE_THRESHOLD}  (tune in anti_spoof_texture.py)")

    print("Loading MTCNN + FaceNet...")
    try:
        mtcnn, facenet = load_mtcnn(), load_facenet()
    except Exception as e:
        print(f"ERROR: {e}"); return 1

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
    print("Running. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None: continue
            frame = cv2.flip(frame, 1)

            detections = detect_faces(frame, mtcnn)
            target = _largest(detections)

            if target is None:
                state.update(None)
            else:
                box = target["box"]

                # ── Liveness gate ─────────────────────────────────────────
                live, score = is_live_texture(frame, box, debug=DEBUG_TEXTURE)
                if not live:
                    draw_spoof_box(frame, box, score)
                    state.update(None)
                    draw_hud(frame, get_today_count(), "EE4001 — Texture Liveness")
                    cv2.imshow(WINDOW_NAME, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"): break
                    continue

                # ── Recognition ───────────────────────────────────────────
                try:
                    emb = get_embedding(target["face_tensor"], facenet)
                except Exception:
                    emb = None

                match = identify_face(emb, known, MATCH_THRESHOLD) if emb is not None else None

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

            draw_hud(frame, get_today_count(), "EE4001 — Texture Liveness")
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        cap.release(); cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
