
from dataclasses import dataclass

import cv2

import mediapipe as mp
from typing import Optional

from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions



@dataclass
class PoseAnalysis:
    has_pose: bool
    landmarks: Optional[list]
    world_landmarks: Optional[list]
    visibility: float
    crop_shape: tuple


class PoseEstimator:
    def __init__(self, model_path="pose_landmarker.task"):
        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def estimate(self, frame, box):
        x1, y1, x2, y2 = box

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return PoseAnalysis(False, None, None, 0, (0,0,0))

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return PoseAnalysis(False, None, None, 0, crop.shape)

        landmarks = result.pose_landmarks[0]

        visibility = sum(
            getattr(lm, "visibility", 0)
            for lm in landmarks
        ) / len(landmarks)

        world_landmarks = (
            result.pose_world_landmarks[0]
            if result.pose_world_landmarks else None
        )
        return PoseAnalysis(True, landmarks, world_landmarks, visibility, crop.shape)

    def close(self):
        self.landmarker.close()

