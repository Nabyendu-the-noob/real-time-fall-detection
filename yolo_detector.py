"""
YOLO-based person detector with ByteTrack.
Layer: Detection + Tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ultralytics import YOLO


@dataclass(frozen=True)
class TrackedBox:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def as_xyxy(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


class YOLODetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        person_class_id: int = 0,
        confidence: float = 0.50,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.model = YOLO(model_path)
        self.person_class_id = person_class_id
        self.confidence = confidence
        self.tracker = tracker

    def detect(self, frame, conf: float | None = None) -> List[TrackedBox]:
        threshold = conf if conf is not None else self.confidence   
        
        results = self.model.track(
            frame,
            classes=[self.person_class_id],
            conf=threshold,                  # ← use threshold, not self.confidence
            iou=0.45,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )[0]

        boxes: List[TrackedBox] = []
        if results.boxes.id is None:
            return boxes

        for raw_box in results.boxes:
            if raw_box.id is None:
                continue
            x1, y1, x2, y2 = map(int, raw_box.xyxy[0].tolist())
            boxes.append(TrackedBox(
                track_id=int(raw_box.id[0]),
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(raw_box.conf[0]),
            ))

        return boxes