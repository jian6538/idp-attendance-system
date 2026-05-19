"""
server_uploader.py
==================
Sends attendance events to the central server.
Works over any network via Cloudflare Tunnel URL.

Usage:
    from server_uploader import upload_attendance, upload_csv_file

    # Send one event (non-blocking, runs in background thread):
    upload_attendance("TAN KAI JIAN", "EEE4948", "2026-05-19 10:51:58")

    # Upload entire CSV file:
    upload_csv_file("attendance_logs/attendance_2026-05-19.csv")
"""

from __future__ import annotations

import csv
import os
import threading
import time
from datetime import datetime
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────
# Change this to your server's tunnel URL or local IP
# Same Wi-Fi example:     "http://192.168.1.20:8000"
# Cloudflare tunnel:       "https://xxxxx.trycloudflare.com"
SERVER_BASE_URL = "https://schemes-attempts-xml-rapidly.trycloudflare.com"

API_KEY    = "device-local-key"
DEVICE_ID  = "raspberry-pi-5"
SOURCE     = "face_recognition"
TIMEOUT    = 15  # seconds
MAX_RETRIES = 3
# ──────────────────────────────────────────────────────────────────────────────

# Try to import requests; if not installed, uploads will be skipped gracefully
try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    print("[server_uploader] WARNING: 'requests' not installed. Server upload disabled.")
    print("  Install it with:  pip install requests")


def _api_url() -> str:
    return f"{SERVER_BASE_URL.rstrip('/')}/api/attendance/events"


def _send_one(student_name: str, course_code: str,
              timestamp: Optional[str] = None) -> bool:
    """Send one attendance event. Returns True on success."""
    if not _HAS_REQUESTS:
        return False

    payload = {
        "student_name": student_name,
        "class_name":   course_code,
        "device_id":    DEVICE_ID,
        "source":       SOURCE,
        "api_key":      API_KEY,
    }
    if timestamp:
        payload["created_at"] = timestamp

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(_api_url(), json=payload, timeout=TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                print(f"[Upload] ✓ {student_name} → {data.get('status', '?')}")
                return True
            else:
                detail = resp.json().get("detail", resp.text)
                print(f"[Upload] ✗ {student_name} → {resp.status_code}: {detail}")
                return False  # don't retry on 4xx errors
        except requests.ConnectionError:
            print(f"[Upload] ✗ {student_name} → connection failed (attempt {attempt}/{MAX_RETRIES})")
        except Exception as e:
            print(f"[Upload] ✗ {student_name} → {e} (attempt {attempt}/{MAX_RETRIES})")

        if attempt < MAX_RETRIES:
            time.sleep(2)

    return False


def upload_attendance(student_name: str, course_code: str,
                      timestamp: Optional[str] = None) -> None:
    """Send attendance to server in a background thread (non-blocking).

    This is the function you call from attendance_logger.py.
    It won't slow down the camera/recognition loop.
    """
    t = threading.Thread(
        target=_send_one,
        args=(student_name, course_code, timestamp),
        daemon=True,
    )
    t.start()


def upload_csv_file(csv_path: str, course_code: str = "") -> dict:
    """Read a daily CSV and upload all rows to the server.

    Returns {"success": N, "errors": N, "skipped": N}
    """
    if not os.path.isfile(csv_path):
        print(f"[Upload] File not found: {csv_path}")
        return {"success": 0, "errors": 0, "skipped": 0}

    success = errors = skipped = 0

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[Upload] Sending {len(rows)} rows from {os.path.basename(csv_path)}...")

    for row in rows:
        name = row.get("Name", "").strip()
        if not name:
            skipped += 1
            continue

        # Build timestamp from Date + Time columns
        date_str = row.get("Date", "").strip()
        time_str = row.get("Time", "").strip()
        ts = f"{date_str} {time_str}" if date_str and time_str else None

        # Use course code from CSV if available, otherwise use parameter
        code = row.get("Course Code", "").strip() or course_code

        if _send_one(name, code, ts):
            success += 1
        else:
            errors += 1

        time.sleep(0.2)  # small delay between requests

    print(f"[Upload] Done: {success} sent, {errors} errors, {skipped} skipped")
    return {"success": success, "errors": errors, "skipped": skipped}


# ── Standalone usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Upload a CSV file:  python server_uploader.py attendance_logs/attendance_2026-05-19.csv
        upload_csv_file(sys.argv[1])
    else:
        print("Usage:")
        print("  python server_uploader.py attendance_logs/attendance_2026-05-19.csv")
        print()
        print("Or import in your code:")
        print("  from server_uploader import upload_attendance")
        print('  upload_attendance("TAN KAI JIAN", "EEE4948")')
