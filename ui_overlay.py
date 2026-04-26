"""
ui_overlay.py
=============
All cv2 drawing for the attendance HUD, bounding boxes, and welcome card.

Everything is drawn IN-PLACE on the BGR frame passed in, and the frame is also
returned for chaining convenience.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette (BGR for OpenCV)
# ---------------------------------------------------------------------------
COLOR_YELLOW = (0, 215, 255)      # scanning
COLOR_GREEN = (60, 200, 80)       # confirmed
COLOR_RED = (60, 60, 230)         # unknown
COLOR_BLUE = (230, 160, 50)       # cooldown
COLOR_ORANGE = (0, 140, 255)      # spoof / fake face detected
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_DARK_PANEL = (30, 30, 30)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _put_text_with_bg(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int] = COLOR_WHITE,
    scale: float = 0.6,
    thickness: int = 1,
    bg_color: Tuple[int, int, int] = COLOR_BLACK,
    bg_alpha: float = 0.55,
    padding: int = 4,
) -> None:
    """Draw text with a semi-transparent dark strip behind it for readability."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    h, w = frame.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - th - padding)
    x2 = min(w, x + tw + padding)
    y2 = min(h, y + baseline + padding)

    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2].copy()
        overlay = np.full_like(roi, bg_color, dtype=np.uint8)
        cv2.addWeighted(overlay, bg_alpha, roi, 1 - bg_alpha, 0, roi)
        frame[y1:y2, x1:x2] = roi

    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _draw_box(frame: np.ndarray, box: List[int], color, thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


# ---------------------------------------------------------------------------
# 1. Scanning (yellow) — confirmation in progress
# ---------------------------------------------------------------------------
def draw_scanning_box(
    frame: np.ndarray,
    box: List[int],
    progress: float,
    label: str = "Scanning...",
) -> np.ndarray:
    _draw_box(frame, box, COLOR_YELLOW, 2)
    x1, y1, x2, y2 = [int(v) for v in box]

    # Label ABOVE box
    _put_text_with_bg(frame, label, (x1, max(18, y1 - 8)),
                      color=COLOR_YELLOW, scale=0.6, thickness=2)

    # Progress bar at the BOTTOM of the bounding box, clamped 0-1
    p = max(0.0, min(1.0, float(progress)))
    bar_h = 6
    bar_y1 = y2 + 4
    bar_y2 = bar_y1 + bar_h
    if bar_y2 < frame.shape[0]:
        # track
        cv2.rectangle(frame, (x1, bar_y1), (x2, bar_y2), (60, 60, 60), -1)
        # fill
        fill_x2 = x1 + int((x2 - x1) * p)
        cv2.rectangle(frame, (x1, bar_y1), (fill_x2, bar_y2), COLOR_YELLOW, -1)

    return frame


# ---------------------------------------------------------------------------
# 2. Confirmed (green)
# ---------------------------------------------------------------------------
def draw_confirmed_box(
    frame: np.ndarray,
    box: List[int],
    name: str,
    matrix_number: str,
) -> np.ndarray:
    _draw_box(frame, box, COLOR_GREEN, 3)
    x1, y1, x2, y2 = [int(v) for v in box]

    _put_text_with_bg(frame, name, (x1, max(18, y1 - 10)),
                      color=COLOR_GREEN, scale=0.7, thickness=2)
    _put_text_with_bg(frame, matrix_number, (x1, min(frame.shape[0] - 4, y2 + 22)),
                      color=COLOR_WHITE, scale=0.6, thickness=1)
    return frame


# ---------------------------------------------------------------------------
# 3. Unknown (red)
# ---------------------------------------------------------------------------
def draw_unknown_box(frame: np.ndarray, box: List[int]) -> np.ndarray:
    _draw_box(frame, box, COLOR_RED, 2)
    x1, y1, _, _ = [int(v) for v in box]
    _put_text_with_bg(frame, "Unknown", (x1, max(18, y1 - 8)),
                      color=COLOR_RED, scale=0.6, thickness=2)
    return frame


# ---------------------------------------------------------------------------
# 4. Cooldown (blue)
# ---------------------------------------------------------------------------
def draw_cooldown_box(frame: np.ndarray, box: List[int], name: str) -> np.ndarray:
    _draw_box(frame, box, COLOR_BLUE, 2)
    x1, y1, x2, y2 = [int(v) for v in box]
    _put_text_with_bg(frame, "Already Recorded", (x1, max(18, y1 - 28)),
                      color=COLOR_BLUE, scale=0.6, thickness=2)
    _put_text_with_bg(frame, name, (x1, max(18, y1 - 8)),
                      color=COLOR_WHITE, scale=0.55, thickness=1)
    return frame


# ---------------------------------------------------------------------------
# 4b. Spoof detected (orange) — printed photo / screen replay rejected
# ---------------------------------------------------------------------------
def draw_spoof_box(frame: np.ndarray, box: List[int], real_prob: float = 0.0) -> np.ndarray:
    """Bright orange box + score when anti-spoofing rejects the face."""
    _draw_box(frame, box, COLOR_ORANGE, 3)
    x1, y1, x2, y2 = [int(v) for v in box]

    _put_text_with_bg(
        frame, "SPOOF DETECTED", (x1, max(18, y1 - 28)),
        color=COLOR_ORANGE, scale=0.65, thickness=2,
    )
    # Show real_prob score so the user can tune the threshold
    _put_text_with_bg(
        frame, f"Live score: {real_prob:.2f}  (need >0.35)",
        (x1, max(18, y1 - 8)),
        color=COLOR_WHITE, scale=0.50, thickness=1,
    )

    # Diagonal cross inside the box
    cv2.line(frame, (x1, y1), (x2, y2), COLOR_ORANGE, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x1, y2), COLOR_ORANGE, 2, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# 5. Welcome card (centre panel, 3s)
# ---------------------------------------------------------------------------
def _draw_check_icon(frame: np.ndarray, center: Tuple[int, int], radius: int = 22) -> None:
    cx, cy = center
    cv2.circle(frame, (cx, cy), radius, COLOR_GREEN, -1)
    cv2.circle(frame, (cx, cy), radius, COLOR_WHITE, 2)
    # Check mark: two strokes
    p1 = (cx - int(radius * 0.45), cy + int(radius * 0.05))
    p2 = (cx - int(radius * 0.10), cy + int(radius * 0.40))
    p3 = (cx + int(radius * 0.50), cy - int(radius * 0.35))
    cv2.line(frame, p1, p2, COLOR_WHITE, 3, cv2.LINE_AA)
    cv2.line(frame, p2, p3, COLOR_WHITE, 3, cv2.LINE_AA)


def draw_welcome_card(
    frame: np.ndarray,
    name: str,
    matrix_number: str,
    timestamp: datetime,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # Card size (centred, not full-screen)
    card_w = min(520, int(w * 0.75))
    card_h = min(260, int(h * 0.60))
    x1 = (w - card_w) // 2
    y1 = (h - card_h) // 2
    x2 = x1 + card_w
    y2 = y1 + card_h

    # Semi-transparent dark panel
    roi = frame[y1:y2, x1:x2].copy()
    overlay = np.full_like(roi, COLOR_DARK_PANEL, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.82, roi, 0.18, 0, roi)
    frame[y1:y2, x1:x2] = roi

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)

    # Green check icon
    _draw_check_icon(frame, (x1 + 50, y1 + 55), radius=22)

    # Heading
    cv2.putText(
        frame,
        "ATTENDANCE RECORDED",
        (x1 + 90, y1 + 62),
        FONT,
        0.78,
        COLOR_GREEN,
        2,
        cv2.LINE_AA,
    )

    # Info lines
    date_str = timestamp.strftime("%d %B %Y")
    time_str = timestamp.strftime("%H:%M:%S")

    lines = [
        f"Name: {name}",
        f"Matrix No: {matrix_number}",
        f"Date: {date_str}",
        f"Time: {time_str}",
        "Status: Present",
    ]

    line_x = x1 + 30
    line_y = y1 + 110
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (line_x, line_y + i * 28),
            FONT,
            0.62,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

    return frame


# ---------------------------------------------------------------------------
# 6. Heads-up display
# ---------------------------------------------------------------------------
def draw_hud(
    frame: np.ndarray,
    total_marked_today: int,
    course_name: str = "EE4001 - Integrated Design Project",
) -> np.ndarray:
    h, w = frame.shape[:2]
    now = datetime.now()

    # Top-left: course + date
    top_left = f"{course_name}   |   {now.strftime('%d %B %Y')}"
    _put_text_with_bg(frame, top_left, (10, 22),
                      color=COLOR_WHITE, scale=0.55, thickness=1)

    # Top-right: live clock
    time_str = now.strftime("%H:%M:%S")
    (tw, _), _ = cv2.getTextSize(time_str, FONT, 0.7, 2)
    _put_text_with_bg(frame, time_str, (w - tw - 14, 26),
                      color=COLOR_YELLOW, scale=0.7, thickness=2)

    # Bottom bar
    bar_h = 34
    y1 = h - bar_h
    roi = frame[y1:h, 0:w].copy()
    overlay = np.full_like(roi, COLOR_BLACK, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, roi)
    frame[y1:h, 0:w] = roi

    status = f"Students Marked Today: {int(total_marked_today)}"
    cv2.putText(frame, status, (12, h - 10), FONT, 0.65,
                COLOR_WHITE, 1, cv2.LINE_AA)

    hint = "Press 'q' to quit"
    (tw2, _), _ = cv2.getTextSize(hint, FONT, 0.55, 1)
    cv2.putText(frame, hint, (w - tw2 - 14, h - 10), FONT, 0.55,
                (180, 180, 180), 1, cv2.LINE_AA)

    return frame
