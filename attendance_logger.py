"""
attendance_logger.py
====================
Daily-CSV attendance writer with course tracking and Present/Late status.

Log file:  attendance_logs/attendance_YYYY-MM-DD.csv
Columns:   Name, Matrix Number, Course Code, Course Name, Date, Time, Status

Status is either "Present" (arrived within grace period) or "Late".
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import List, Optional

from server_uploader import upload_attendance

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance_logs")

CSV_HEADERS = [
    "Name", "Matrix Number", "Course Code", "Course Name",
    "Date", "Time", "Status",
]


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _today_csv_path() -> str:
    return os.path.join(LOG_DIR, f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv")


def _read_today_rows() -> List[dict]:
    path = _today_csv_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def already_marked_today(matrix_number: str,
                         course_code: Optional[str] = None) -> bool:
    """True if this student is already logged today (optionally for a specific course)."""
    for row in _read_today_rows():
        if row.get("Matrix Number", "").strip() != matrix_number.strip():
            continue
        if course_code is None:
            return True
        if row.get("Course Code", "").strip() == course_code.strip():
            return True
    return False


def get_today_count(course_code: Optional[str] = None) -> int:
    """Number of rows today, optionally filtered by course."""
    rows = _read_today_rows()
    if course_code is None:
        return len(rows)
    return sum(
        1 for r in rows
        if r.get("Course Code", "").strip() == course_code.strip()
    )


def log_attendance(
    name: str,
    matrix_number: str,
    status: str = "Present",
    course_code: str = "",
    course_name: str = "",
) -> bool:
    """Append one attendance row. Returns False if duplicate for same course today."""
    _ensure_log_dir()

    if already_marked_today(matrix_number, course_code or None):
        return False

    now  = datetime.now()
    path = _today_csv_path()
    new_file = not os.path.isfile(path)

    row = {
        "Name":          name,
        "Matrix Number": matrix_number,
        "Course Code":   course_code,
        "Course Name":   course_name,
        "Date":          now.strftime("%Y-%m-%d"),
        "Time":          now.strftime("%H:%M:%S"),
        "Status":        status,
    }

    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    except OSError:
        return False

    # Send to central server in background (non-blocking)
    upload_attendance(name, course_code, now.strftime("%Y-%m-%d %H:%M:%S"))

    return True
