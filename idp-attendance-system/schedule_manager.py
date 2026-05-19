"""
schedule_manager.py
===================
Reads admin/timetable.csv and admin/students/<COURSE_CODE>.csv.

Key responsibilities:
  - Find the currently active course based on today's day + current time
  - Determine attendance status: "Present" (within grace period) or "Late"
  - Load the enrolled student list for any course

Timetable CSV columns:
    course_code, course_name, day, start_time, end_time

Student CSV columns (one file per course in admin/students/):
    name, matrix_number
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, time as dtime
from typing import Dict, List, Optional

HERE           = os.path.dirname(os.path.abspath(__file__))
ADMIN_DIR      = os.path.join(HERE, "admin")
TIMETABLE_PATH = os.path.join(ADMIN_DIR, "timetable.csv")
STUDENTS_DIR   = os.path.join(ADMIN_DIR, "students")

GRACE_MINUTES  = 10   # within this many minutes of start = "Present"; after = "Late"

# Map CSV day names → Python weekday numbers (Monday=0 … Sunday=6)
DAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class CourseSlot:
    """One row from timetable.csv."""
    def __init__(self, code: str, name: str, day: int,
                 start: dtime, end: dtime) -> None:
        self.course_code = code
        self.course_name = name
        self.day         = day          # 0=Mon … 6=Sun
        self.start_time  = start
        self.end_time    = end

    def __repr__(self) -> str:
        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        return (f"<CourseSlot {self.course_code} {days[self.day]} "
                f"{self.start_time.strftime('%H:%M')}-"
                f"{self.end_time.strftime('%H:%M')}>")


# ---------------------------------------------------------------------------
# Timetable loading
# ---------------------------------------------------------------------------

def load_timetable() -> List[CourseSlot]:
    """Read admin/timetable.csv.  Returns [] if file is missing or corrupt."""
    slots: List[CourseSlot] = []
    if not os.path.isfile(TIMETABLE_PATH):
        return slots
    try:
        with open(TIMETABLE_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    day_int = DAY_MAP.get(row["day"].strip().lower())
                    if day_int is None:
                        continue
                    start = dtime(*[int(x) for x in row["start_time"].strip().split(":")])
                    end   = dtime(*[int(x) for x in row["end_time"].strip().split(":")])
                    slots.append(CourseSlot(
                        code  = row["course_code"].strip(),
                        name  = row["course_name"].strip(),
                        day   = day_int,
                        start = start,
                        end   = end,
                    ))
                except (KeyError, ValueError):
                    continue
    except OSError:
        pass
    return slots


def save_timetable(slots: List[CourseSlot]) -> None:
    """Overwrite admin/timetable.csv with the given slots."""
    os.makedirs(ADMIN_DIR, exist_ok=True)
    days_rev = {v: k.capitalize() for k, v in DAY_MAP.items()}
    with open(TIMETABLE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["course_code","course_name","day","start_time","end_time"])
        for s in slots:
            writer.writerow([
                s.course_code,
                s.course_name,
                days_rev[s.day],
                s.start_time.strftime("%H:%M"),
                s.end_time.strftime("%H:%M"),
            ])


# ---------------------------------------------------------------------------
# Active course detection
# ---------------------------------------------------------------------------

def get_current_course(slots: Optional[List[CourseSlot]] = None,
                       now: Optional[datetime] = None) -> Optional[CourseSlot]:
    """Return the CourseSlot that is currently running, or None.

    'Currently running' means: today's weekday matches the slot's day AND
    current time is between start_time and end_time (inclusive).
    """
    if slots is None:
        slots = load_timetable()
    if now is None:
        now = datetime.now()

    today   = now.weekday()
    current = now.time().replace(second=0, microsecond=0)

    for slot in slots:
        if slot.day == today and slot.start_time <= current <= slot.end_time:
            return slot
    return None


def get_attendance_status(slot: CourseSlot,
                          arrival: Optional[datetime] = None) -> str:
    """Return 'Present' if arrival is within GRACE_MINUTES of slot start, else 'Late'.

    arrival defaults to datetime.now().
    """
    if arrival is None:
        arrival = datetime.now()

    start_dt = datetime.combine(arrival.date(), slot.start_time)
    delta_minutes = (arrival - start_dt).total_seconds() / 60.0

    if delta_minutes <= GRACE_MINUTES:
        return "Present"
    return "Late"


# ---------------------------------------------------------------------------
# Student list management
# ---------------------------------------------------------------------------

def get_students_path(course_code: str) -> str:
    return os.path.join(STUDENTS_DIR, f"{course_code}.csv")


def load_enrolled_students(course_code: str) -> List[Dict[str, str]]:
    """Return list of {name, matrix_number} for the given course.

    Returns [] if the CSV doesn't exist.
    """
    path = get_students_path(course_code)
    if not os.path.isfile(path):
        return []
    students: List[Dict[str, str]] = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("name", "").strip()
                mn   = row.get("matrix_number", "").strip()
                if name and mn:
                    students.append({"name": name, "matrix_number": mn})
    except OSError:
        pass
    return students


def save_enrolled_students(course_code: str,
                           students: List[Dict[str, str]]) -> None:
    """Overwrite admin/students/<course_code>.csv."""
    os.makedirs(STUDENTS_DIR, exist_ok=True)
    path = get_students_path(course_code)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "matrix_number"])
        writer.writeheader()
        writer.writerows(students)


def is_enrolled(matrix_number: str, course_code: str) -> bool:
    """True if the student is in the course's CSV."""
    students = load_enrolled_students(course_code)
    return any(s["matrix_number"].lower() == matrix_number.lower()
               for s in students)


def list_all_courses() -> List[str]:
    """Return sorted list of course codes that have a student CSV."""
    if not os.path.isdir(STUDENTS_DIR):
        return []
    return sorted(
        f[:-4] for f in os.listdir(STUDENTS_DIR)
        if f.endswith(".csv")
    )
