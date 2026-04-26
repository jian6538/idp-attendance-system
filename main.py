"""
main.py
=======
Door-camera attendance system — with timetable-aware course detection.

On startup:
  1. Loads MTCNN + FaceNet models
  2. Reads admin/timetable.csv to find today's active course
  3. Loads ONLY the students enrolled in that course
  4. Marks attendance as Present / Late based on 10-min grace period

If no course is running right now the system shows a warning overlay and
waits — it will not mark anyone until a scheduled slot is active.

Press 'q' to quit.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import cv2

from anti_spoof import load_anti_spoof_model, DEFAULT_LIVENESS_THRESHOLD
from attendance_logger import get_today_count, log_attendance
from face_utils import (
    DEFAULT_MATCH_THRESHOLD,
    check_liveness,
    detect_faces,
    get_embedding,
    identify_face,
    load_all_embeddings,
    load_facenet,
    load_mtcnn,
)
from recognition_state import RecognitionState
from schedule_manager import (
    get_attendance_status,
    get_current_course,
    load_enrolled_students,
    load_timetable,
)
from ui_overlay import (
    draw_confirmed_box,
    draw_cooldown_box,
    draw_hud,
    draw_scanning_box,
    draw_spoof_box,
    draw_unknown_box,
    draw_welcome_card,
)

HERE       = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(HERE, "database", "students.json")
EMB_DIR    = os.path.join(HERE, "embeddings")
AS_WEIGHTS = os.path.join(HERE, "anti_spoof_weights", "AntiSpoofing_bin_1.5_128.onnx")

WINDOW_NAME        = "Attendance System"
CAMERA_INDEX       = 0
FRAME_W, FRAME_H   = 640, 480
WELCOME_CARD_SECS  = 3.0
CONFIRM_DURATION   = 2.5
COOLDOWN_MINUTES   = 60
MATCH_THRESHOLD    = DEFAULT_MATCH_THRESHOLD   # 0.82
LIVENESS_THRESHOLD = 0.5    # hairymax binary model: real_prob >= 0.5 → live
                             # set to None to disable liveness check

# How often (seconds) to re-check timetable for course changes mid-session
TIMETABLE_REFRESH_INTERVAL = 60


def _largest_face(detections: List[Dict]) -> Optional[Dict]:
    if not detections:
        return None
    return max(
        detections,
        key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]),
    )


def _open_camera() -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return cap


def _draw_no_course_overlay(frame, msg: str = "") -> None:
    """Grey overlay shown when no course is currently scheduled."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "NO CLASS IN SESSION",
                (w // 2 - 200, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2, cv2.LINE_AA)
    if msg:
        cv2.putText(frame, msg,
                    (w // 2 - 180, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1, cv2.LINE_AA)


def _enrolled_matrix_numbers(course_code: str) -> set:
    """Set of matrix numbers enrolled in the course (for fast lookup)."""
    return {s["matrix_number"] for s in load_enrolled_students(course_code)}


def main() -> int:
    # ── Models ────────────────────────────────────────────────────────────
    print("Loading MTCNN + FaceNet (CPU)...")
    try:
        mtcnn   = load_mtcnn()
        facenet = load_facenet()
    except Exception as e:
        print(f"ERROR: {e}"); return 1
    print("Face models loaded.")

    print("Loading anti-spoofing model...")
    anti_spoof_model = load_anti_spoof_model(AS_WEIGHTS)
    if anti_spoof_model:
        print("  → Liveness ENABLED (MiniFASNetV2)")
    else:
        print("  → Liveness DISABLED")

    # ── Face embeddings (all enrolled students) ────────────────────────────
    print("Loading face embeddings...")
    all_known = load_all_embeddings(DB_PATH, EMB_DIR)
    print(f"  {len(all_known)} total enrolled student(s) in database.")

    # ── Timetable ──────────────────────────────────────────────────────────
    timetable   = load_timetable()
    last_refresh = time.time()

    current_slot     = get_current_course(timetable)
    enrolled_set     = _enrolled_matrix_numbers(current_slot.course_code) if current_slot else set()
    # Filter known embeddings to only those enrolled in current course
    known = [k for k in all_known if k["matrix_number"] in enrolled_set] if current_slot else []

    if current_slot:
        print(f"  Active course: {current_slot.course_code} — {current_slot.course_name}")
        print(f"  Enrolled students in this course: {len(known)}")
    else:
        print("  No course scheduled right now — waiting for a timetable slot.")

    state = RecognitionState(CONFIRM_DURATION, COOLDOWN_MINUTES)

    cap = _open_camera()
    if cap is None:
        print(f"ERROR: cannot open camera {CAMERA_INDEX}."); return 1

    welcome_card_data: Optional[Dict] = None
    welcome_card_t:    float          = 0.0

    print("System running. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.flip(frame, 1)

            # ── Refresh timetable periodically ────────────────────────────
            if time.time() - last_refresh > TIMETABLE_REFRESH_INTERVAL:
                timetable    = load_timetable()
                new_slot     = get_current_course(timetable)
                if (new_slot is None) != (current_slot is None) or (
                    new_slot and current_slot and
                    new_slot.course_code != current_slot.course_code
                ):
                    # Course changed — reload enrolled students
                    current_slot = new_slot
                    enrolled_set = _enrolled_matrix_numbers(current_slot.course_code) if current_slot else set()
                    known        = [k for k in all_known if k["matrix_number"] in enrolled_set] if current_slot else []
                    state        = RecognitionState(CONFIRM_DURATION, COOLDOWN_MINUTES)
                    if current_slot:
                        print(f"[Timetable] Switched to {current_slot.course_code} — {current_slot.course_name}")
                    else:
                        print("[Timetable] No course active.")
                last_refresh = time.time()

            # ── No course running ─────────────────────────────────────────
            if current_slot is None:
                _draw_no_course_overlay(frame, "Check admin/timetable.csv")
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            course_label = f"{current_slot.course_code} — {current_slot.course_name}"

            # ── Face detection ────────────────────────────────────────────
            detections = detect_faces(frame, mtcnn)
            target     = _largest_face(detections)

            if target is None:
                state.update(None)
            else:
                box = target["box"]

                # ── Liveness gate (optional) ───────────────────────────────
                if LIVENESS_THRESHOLD is not None:
                    live, real_prob = check_liveness(
                        frame, box, anti_spoof_model, LIVENESS_THRESHOLD)
                    if not live:
                        draw_spoof_box(frame, box, real_prob)
                        state.update(None)
                        draw_hud(frame, get_today_count(current_slot.course_code), course_label)
                        cv2.imshow(WINDOW_NAME, frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        continue

                # ── Recognition ───────────────────────────────────────────
                try:
                    emb = get_embedding(target["face_tensor"], facenet)
                except Exception:
                    emb = None

                match = identify_face(emb, known, MATCH_THRESHOLD) if emb is not None else None

                if match is None:
                    draw_unknown_box(frame, box)
                    state.update(None)
                else:
                    matrix_number = match["matrix_number"]
                    name          = match["name"]

                    # Double-check student is enrolled in this course
                    if matrix_number not in enrolled_set:
                        draw_unknown_box(frame, box)
                        state.update(None)
                    elif state.is_on_cooldown(matrix_number):
                        draw_cooldown_box(frame, box, name)
                        state.update(None)
                    else:
                        status = state.update(matrix_number)

                        if status == "confirmed":
                            arrival_status = get_attendance_status(current_slot)
                            written = log_attendance(
                                name          = name,
                                matrix_number = matrix_number,
                                status        = arrival_status,
                                course_code   = current_slot.course_code,
                                course_name   = current_slot.course_name,
                            )
                            state.mark_done(matrix_number)
                            if written:
                                welcome_card_data = {
                                    "name":          name,
                                    "matrix_number": matrix_number,
                                    "timestamp":     datetime.now(),
                                    "status":        arrival_status,
                                }
                                welcome_card_t = time.time()
                            draw_confirmed_box(frame, box, name, matrix_number)

                        elif status == "confirming":
                            draw_scanning_box(frame, box, state.get_progress(),
                                              f"Scanning... {name}")
                        else:
                            draw_scanning_box(frame, box, 0.0, "Scanning...")

            # ── Welcome card ──────────────────────────────────────────────
            if welcome_card_data is not None:
                if time.time() - welcome_card_t < WELCOME_CARD_SECS:
                    draw_welcome_card(
                        frame,
                        welcome_card_data["name"],
                        welcome_card_data["matrix_number"],
                        welcome_card_data["timestamp"],
                    )
                else:
                    welcome_card_data = None

            draw_hud(frame, get_today_count(current_slot.course_code), course_label)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
