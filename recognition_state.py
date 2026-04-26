"""
recognition_state.py
====================
Confirmation timer + per-student cooldown.

The door camera sees a student walking up.  We don't want to log attendance on
a single flickering frame — we want the same student recognized consistently
for `confirm_duration` seconds before we commit.  After marking, the student is
placed on a `cooldown_minutes` hold so walking back out and in again doesn't
double-log.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, Optional


class RecognitionState:
    def __init__(
        self,
        confirm_duration: float = 2.5,
        cooldown_minutes: int = 60,
    ) -> None:
        self.confirm_duration: float = float(confirm_duration)
        self.cooldown_minutes: int = int(cooldown_minutes)

        self.current_candidate: Optional[str] = None
        self.candidate_start_time: float = 0.0

        # matrix_number -> datetime of last successful attendance mark
        self.last_marked: Dict[str, datetime] = {}

    # ------------------------------------------------------------------ #
    # Core state transitions
    # ------------------------------------------------------------------ #
    def update(self, matrix_number: Optional[str]) -> str:
        """Advance the confirmation state machine.

        Returns one of:
            "confirming" — same candidate still accumulating time
            "confirmed"  — hit confirm_duration; caller should log attendance
            "reset"      — no candidate, or candidate changed; timer reset
        """
        now = time.time()

        # No face / no identification this frame -> abandon any pending candidate.
        if matrix_number is None:
            self._reset_candidate()
            return "reset"

        # New or different candidate -> start a fresh timer.
        if self.current_candidate != matrix_number:
            self.current_candidate = matrix_number
            self.candidate_start_time = now
            return "reset"

        # Same candidate as last frame.
        elapsed = now - self.candidate_start_time
        if elapsed >= self.confirm_duration:
            return "confirmed"
        return "confirming"

    # ------------------------------------------------------------------ #
    # Cooldown logic
    # ------------------------------------------------------------------ #
    def is_on_cooldown(self, matrix_number: str) -> bool:
        """True if this student was marked within the cooldown window."""
        last = self.last_marked.get(matrix_number)
        if last is None:
            return False
        return datetime.now() - last < timedelta(minutes=self.cooldown_minutes)

    def mark_done(self, matrix_number: str) -> None:
        """Record the successful attendance and clear the pending candidate.

        Clearing the candidate immediately means the state machine is ready
        for the NEXT student in a back-to-back door queue.
        """
        self.last_marked[matrix_number] = datetime.now()
        self._reset_candidate()

    # ------------------------------------------------------------------ #
    # UI helper
    # ------------------------------------------------------------------ #
    def get_progress(self) -> float:
        """Confirmation progress in [0.0, 1.0] for the scanning progress bar."""
        if self.current_candidate is None or self.confirm_duration <= 0:
            return 0.0
        elapsed = time.time() - self.candidate_start_time
        return max(0.0, min(1.0, elapsed / self.confirm_duration))

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _reset_candidate(self) -> None:
        self.current_candidate = None
        self.candidate_start_time = 0.0
