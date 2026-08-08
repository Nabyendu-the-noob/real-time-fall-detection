"""
CSV logger + screenshot persistence for confirmed fall events.
Layer: Logging
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

import cv2


class FallLogger:
    def __init__(self, log_file: str = "fall_events.csv", screenshot_dir: str = "fall_logs") -> None:
        self.log_file = log_file
        self.screenshot_dir = screenshot_dir
        os.makedirs(self.screenshot_dir, exist_ok=True)

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp",
                    "Person_ID",
                    "Severity",
                    "Fall_Duration_sec",
                    "Fall_Score",
                    "Screenshot_Path",
                ])

        self.logged_events: set[tuple[int, str]] = set()

    def log(self, frame, person_id: int, severity: str, duration: float, score: float | None = None) -> None:
        now = datetime.now()
        minute_key = (person_id, now.strftime("%Y%m%d%H%M"))
        if minute_key in self.logged_events:
            return

        self.logged_events.add(minute_key)

        filename = f"fall_{now.strftime('%Y%m%d_%H%M%S')}_person{person_id}.png"
        screenshot_path = os.path.join(self.screenshot_dir, filename)
        cv2.imwrite(screenshot_path, frame)

        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                now.strftime("%Y-%m-%d %H:%M:%S"),
                person_id,
                severity,
                round(float(duration), 1),
                None if score is None else round(float(score), 3),
                screenshot_path,
            ])

        msg = f"[Logger] Fall logged: Person {person_id} | {severity} | {duration:.1f}s"
        if score is not None:
            msg += f" | score={score:.2f}"
        print(msg)
