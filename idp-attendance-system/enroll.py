"""
enroll.py
=========
Enrol a new student into the attendance database.

Usage:
    python enroll.py

Prompts for Name and Matrix Number, opens the webcam, captures 15 valid face
crops, averages their embeddings, saves them to embeddings/<matrix>.npy, and
appends a record to database/students.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List

import cv2
import numpy as np

from face_utils import (
    detect_faces,
    get_embedding,
    load_facenet,
    load_mtcnn,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(HERE, "database")
DB_PATH = os.path.join(DB_DIR, "students.json")
EMB_DIR = os.path.join(HERE, "embeddings")

TARGET_FRAMES = 15
WINDOW_NAME = "Enrolment - press 'q' to cancel"


def _ensure_dirs() -> None:
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)


def _load_students() -> List[dict]:
    if not os.path.isfile(DB_PATH):
        return []
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_students(students: List[dict]) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(students, f, indent=2, ensure_ascii=False)


def _prompt(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _largest_face(detections):
    """Pick the detection with the largest bounding-box area."""
    if not detections:
        return None
    return max(
        detections,
        key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]),
    )


def main() -> int:
    _ensure_dirs()

    print("=== Student Enrolment ===")
    name = _prompt("Enter student name: ")
    matrix_number = _prompt("Enter matrix number: ")

    if not name or not matrix_number:
        print("ERROR: Name and matrix number are required.")
        return 1

    students = _load_students()
    for s in students:
        if s.get("matrix_number", "").lower() == matrix_number.lower():
            overwrite = _prompt(
                f"Matrix number '{matrix_number}' already exists. Overwrite? (y/N): "
            ).lower()
            if overwrite != "y":
                print("Aborted.")
                return 1
            students = [s for s in students if s.get("matrix_number") != matrix_number]
            break

    print("Loading models (this takes a few seconds)...")
    mtcnn = load_mtcnn()
    facenet = load_facenet()
    print("Models loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (device 0).")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Look at the camera. Capturing 15 frames...")

    collected: List[np.ndarray] = []
    last_capture_t = 0.0
    min_gap = 0.15  # seconds between captures, to diversify samples

    try:
        while len(collected) < TARGET_FRAMES:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.flip(frame, 1)

            detections = detect_faces(frame, mtcnn)
            target = _largest_face(detections)

            display = frame.copy()

            if target is not None:
                x1, y1, x2, y2 = target["box"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (60, 200, 80), 2)
                cv2.putText(
                    display,
                    f"{len(collected)}/{TARGET_FRAMES}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (60, 200, 80),
                    2,
                    cv2.LINE_AA,
                )
                now = time.time()
                if now - last_capture_t >= min_gap:
                    try:
                        emb = get_embedding(target["face_tensor"], facenet)
                        collected.append(emb)
                        last_capture_t = now
                        print(f"Captured {len(collected)}/{TARGET_FRAMES} frames...")
                    except Exception as e:
                        print(f"  skip (embedding failed: {e})")
            else:
                cv2.putText(
                    display,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (60, 60, 230),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_NAME, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Cancelled by user.")
                return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(collected) < TARGET_FRAMES:
        print("ERROR: did not collect enough frames.")
        return 1

    avg = np.mean(np.stack(collected, axis=0), axis=0).astype(np.float32)
    n = np.linalg.norm(avg)
    if n > 0:
        avg = avg / n

    emb_path = os.path.join(EMB_DIR, f"{matrix_number}.npy")
    np.save(emb_path, avg)

    students.append(
        {
            "name": name,
            "matrix_number": matrix_number,
            "embedding_path": os.path.relpath(emb_path, HERE).replace("\\", "/"),
        }
    )
    _save_students(students)

    print(f"\n\u2713 Enrolled {name} ({matrix_number})")
    print(f"  Embedding: {emb_path}")
    print(f"  Database : {DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
