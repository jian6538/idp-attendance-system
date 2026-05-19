"""
main_facenox.py  —  Option 1: Passive liveness via facenox ONNX model
======================================================================
Setup:
    pip install onnxruntime
    python download_weights_facenox.py
    python main_facenox.py
"""

from __future__ import annotations

import os, sys, time
from datetime import datetime
from typing import Dict, List, Optional

import cv2

from anti_spoof_facenox import load_facenox_model, is_live_facenox, LIVENESS_THR
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

HERE        = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(HERE, "database", "students.json")
EMB_DIR     = os.path.join(HERE, "embeddings")
WINDOW_NAME = "Attendance — Facenox Liveness"
CAMERA_IDX  = 0
FRAME_W, FRAME_H      = 640, 480
WELCOME_CARD_SECS     = 3.0
CONFIRM_DURATION      = 2.5
COOLDOWN_MINUTES      = 60
MATCH_THRESHOLD       = DEFAULT_MATCH_THRESHOLD
LIVENESS_THRESHOLD    = LIVENESS_THR   # 0.55


def _largest(detections):
    return max(detections, key=lambda d: (d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1])) if detections else None


def main() -> int:
    print("=== Facenox Liveness Mode ===")
    print("Loading MTCNN + FaceNet...")
    try:
        mtcnn, facenet = load_mtcnn(), load_facenet()
    except Exception as e:
        print(f"ERROR: {e}"); return 1

    print("Loading facenox anti-spoof model...")
    spoof_model = load_facenox_model()
    if spoof_model:
        print("  → Liveness ENABLED (facenox MiniFASNetV2-SE, 128×128)")
    else:
        print("  → Liveness DISABLED (run download_weights_facenox.py first)")

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
                live, score = is_live_facenox(frame, box, spoof_model, LIVENESS_THRESHOLD)
                if not live:
                    draw_spoof_box(frame, box, score)
                    state.update(None)
                    draw_hud(frame, get_today_count(), "EE4001 — Facenox Liveness")
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

            draw_hud(frame, get_today_count(), "EE4001 — Facenox Liveness")
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        cap.release(); cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
