

import math
from dataclasses import dataclass
from collections import deque
from typing import Optional

import numpy as np

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24


@dataclass
class FeatureVector:
    torso_angle: float
    aspect_ratio: float
    hip_velocity: float
    upward_hip_velocity: float
    height_drop: float
    com_shift: float
    angle_delta: float
    persistence: float
    visibility: float
    height_collapse: float
    width_expansion: float
    smoothed_score_hint: float
    
    head_velocity: float          # downward speed of head
    shoulder_hip_separation: float  # vertical gap between shoulders and hips
    knee_collapse_left: float     # left knee angle deviation from 180°
    knee_collapse_right: float    # right knee angle deviation from 180°
    bilateral_asymmetry: float    # left vs right side motion difference
    post_impact_stillness: float  # near-zero motion after a high-score spike
    torso_tilt_3d: float          # 3D angle from world landmarks
    floor_proximity: float


class FeatureExtractor:
    def __init__(self, history_size: int = 10, ema_alpha: float = 0.3):
        self.history_size = history_size
        self.ema_alpha = ema_alpha

    @staticmethod
    def clamp01(v):
        return max(0.0, min(1.0, float(v)))

    @staticmethod
    def normalize(v, low, high):
        if high <= low:
            return 0.0
        return max(0.0, min(1.0, (v - low) / (high - low)))

    def calculate_torso_angle(self, landmarks, h, w):
        ls = landmarks[L_SHOULDER]
        rs = landmarks[R_SHOULDER]
        lh = landmarks[L_HIP]
        rh = landmarks[R_HIP]

        shoulder_center = (
            ((ls.x + rs.x) / 2.0) * w,
            ((ls.y + rs.y) / 2.0) * h,
        )

        hip_center = (
            ((lh.x + rh.x) / 2.0) * w,
            ((lh.y + rh.y) / 2.0) * h,
        )

        dy = shoulder_center[1] - hip_center[1]
        dx = shoulder_center[0] - hip_center[0]

        angle_from_vertical = abs(
            np.degrees(
                np.arctan2(abs(dx), abs(dy))
            )
        )
        return angle_from_vertical
    
    def update_score_history(self, state, final_score: float) -> None:
        history = state.setdefault("score_history", deque(maxlen=self.history_size))
        history.append(final_score)

    def _ema(self, prev, current):
        if prev is None:
            return current
        return self.ema_alpha * current + (1 - self.ema_alpha) * prev

    # Compute angle at the knee joint from 3 landmarks
    @staticmethod
    def _joint_angle(a, b, c, h, w):
        """Angle at point b, between vectors b→a and b→c. Returns degrees."""
        ax, ay = a.x * w, a.y * h
        bx, by = b.x * w, b.y * h
        cx, cy = c.x * w, c.y * h
        v1 = (ax - bx, ay - by)
        v2 = (cx - bx, cy - by)
        mag1 = (v1[0]**2 + v1[1]**2) ** 0.5
        mag2 = (v2[0]**2 + v2[1]**2) ** 0.5
        if mag1 < 1e-6 or mag2 < 1e-6:
            return 180.0
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_a))

    # Compute 3D torso tilt from world landmarks
    @staticmethod
    def _torso_tilt_3d(world_landmarks):
        """
        Angle of torso from vertical using metric 3D coords.
        World landmarks are in metres, hip-centred.
        """
        if world_landmarks is None:
            return 0.0
        ls = world_landmarks[11]
        rs = world_landmarks[12]
        lh = world_landmarks[23]
        rh = world_landmarks[24]
        sx = (ls.x + rs.x) / 2
        sy = (ls.y + rs.y) / 2
        sz = (ls.z + rs.z) / 2
        hx = (lh.x + rh.x) / 2
        hy = (lh.y + rh.y) / 2
        hz = (lh.z + rh.z) / 2
        dx, dy, dz = sx - hx, sy - hy, sz - hz
        horiz = (dx**2 + dz**2) ** 0.5
        return abs(math.degrees(math.atan2(horiz, abs(dy))))
    
    def extract(self, state, box, pose_result):
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        
        baseline_h = state.setdefault("baseline_h", bh)
        baseline_w = state.setdefault("baseline_w", bw)
        
        if not state.get("fall_detected", False):
            alpha = 0.05   # ~20 frames to adapt; too slow to absorb a fall
            state["baseline_h"] = (1 - alpha) * state["baseline_h"] + alpha * bh
            state["baseline_w"] = (1 - alpha) * state["baseline_w"] + alpha * bw
            baseline_h = state["baseline_h"]
            baseline_w = state["baseline_w"]

        height_collapse = max(0, 1 - (bh / baseline_h))

        width_expansion = max(0, (bw / baseline_w) - 1)

        aspect_ratio = bw / bh

        if not pose_result.has_pose:
            return FeatureVector(
                torso_angle=state.get("last_torso_angle", 0),
                aspect_ratio=aspect_ratio,
                hip_velocity=state.get("last_hip_velocity", 0),
                upward_hip_velocity=state.get("last_upward_hip_velocity", 0),
                height_drop=state.get("last_height_drop", 0),
                com_shift=0,
                angle_delta=0,
                persistence=0,
                visibility=state.get("last_visibility", 0.7),
                smoothed_score_hint=0,
                height_collapse=height_collapse,
                width_expansion=width_expansion,
                head_velocity=state.get("head_velocity", 0),
                shoulder_hip_separation=state.get("shoulder_hip_separation", 0.5),
                knee_collapse_left=0,
                knee_collapse_right=0,
                bilateral_asymmetry=0,
                post_impact_stillness=0,
                torso_tilt_3d=state.get("torso_tilt_3d", 0),
                floor_proximity=self.clamp01((y2 / state.get("frame_h", 1) - 0.3) / 0.7),
            )

        landmarks = pose_result.landmarks
        h, w = pose_result.crop_shape[:2]

        visibility = pose_result.visibility
        state["last_visibility"] = visibility

        torso_angle = self.calculate_torso_angle(landmarks, h, w)

        lh = landmarks[L_HIP]
        rh = landmarks[R_HIP]

        hip_x = ((lh.x + rh.x) / 2.0) * w
        hip_y_global = y1 + ((lh.y + rh.y)/2)*h

        prev_hip = state.get("prev_hip")

        hip_velocity = 0
        upward_hip_velocity = 0.0
        height_drop = 0
        com_shift = 0
        
        baseline_hip_y = state.setdefault("baseline_hip_y", hip_y_global)
        
        if not state.get("fall_detected", False):
            alpha = 0.008
            state["baseline_hip_y"] = (1 - alpha) * baseline_hip_y + alpha * hip_y_global
            baseline_hip_y = state["baseline_hip_y"]

        if prev_hip is not None:
            dx = hip_x - prev_hip[0]
            dy = hip_y_global - prev_hip[1]
        
            com_shift = (dx**2 + dy**2) ** 0.5 / bh
            hip_velocity = max(0, dy / bh)
            upward_hip_velocity = max(0, -dy / bh)
            height_drop = max(
                0,
                (hip_y_global - baseline_hip_y) / bh
            )

        state["prev_hip"] = (hip_x, hip_y_global)
        
        state["last_hip_velocity"] = hip_velocity
        state["last_upward_hip_velocity"] = upward_hip_velocity
        state["last_height_drop"] = height_drop
        state["last_torso_angle"] = torso_angle

        prev_angle = state.get("prev_angle")
        angle_delta = abs(torso_angle - prev_angle) if prev_angle is not None else 0
        state["prev_angle"] = torso_angle

        smoothed_angle = self._ema(state.get("smoothed_angle"), torso_angle)
        state["smoothed_angle"] = smoothed_angle

        angle_score = self.normalize(smoothed_angle, 20, 80)
        velocity_score = self.normalize(hip_velocity, 0.02, 0.35)
        ratio_score = self.normalize(aspect_ratio, 0.55, 1.2)
        drop_score = self.normalize(height_drop, 0.01, 0.25)
        delta_score = self.normalize(angle_delta, 5, 45)

        score_hint = (
            angle_score * 0.20
            + velocity_score * 0.15
            + drop_score * 0.15
            + ratio_score * 0.15
            + delta_score * 0.10
        )

        history = state.setdefault("score_history", deque(maxlen=self.history_size))
        #history.append(score_hint)

        persistence = sum(1 for s in history if s > 0.55) / max(1, len(history))
        
        # ── Head velocity ──────────────────────────────────────
        HEAD_IDX = [0, 7, 8]  # nose, left ear, right ear
        head_y_global = 0.0
        head_velocity = 0.0
        if pose_result.has_pose:
            ys = [pose_result.landmarks[i].y for i in HEAD_IDX
                if i < len(pose_result.landmarks)]
            if ys:
                head_y_global = y1 + (sum(ys) / len(ys)) * h
                prev_head_y = state.get("prev_head_y")
                if prev_head_y is not None:
                    head_velocity = max(0, (head_y_global - prev_head_y) / bh)
                state["prev_head_y"] = head_y_global

        # ── Shoulder-hip vertical separation ──────────────────
        shoulder_hip_separation = 0.0
        if pose_result.has_pose:
            ls = pose_result.landmarks[11]
            rs = pose_result.landmarks[12]
            shoulder_y_global = y1 + ((ls.y + rs.y) / 2) * h
            # Negative/zero = shoulders at or below hip level (fallen)
            # Normalized by box height
            shoulder_hip_separation = max(
                0,
                (hip_y_global - shoulder_y_global) / bh
            )
            # 0 = collapsed, ~0.5 = normal standing

        # ── Knee collapse ──────────────────────────────────────
        knee_collapse_left  = 0.0
        knee_collapse_right = 0.0
        if pose_result.has_pose and len(pose_result.landmarks) > 28:
            lms = pose_result.landmarks
            left_angle  = self._joint_angle(lms[23], lms[25], lms[27], h, w)
            right_angle = self._joint_angle(lms[24], lms[26], lms[28], h, w)
            # 180° = straight leg. Collapse score rises as knee buckles
            knee_collapse_left  = self.clamp01((180 - left_angle)  / 90)
            knee_collapse_right = self.clamp01((180 - right_angle) / 90)

        # ── Bilateral asymmetry ────────────────────────────────
        bilateral_asymmetry = 0.0
        if pose_result.has_pose:
            lh_lm = pose_result.landmarks[23]
            rh_lm = pose_result.landmarks[24]
            lh_y_global = y1 + lh_lm.y * h
            rh_y_global = y1 + rh_lm.y * h
            prev_lhy = state.get("prev_left_hip_y")
            prev_rhy = state.get("prev_right_hip_y")
            if prev_lhy is not None and prev_rhy is not None:
                lv = abs(lh_y_global - prev_lhy)
                rv = abs(rh_y_global - prev_rhy)
                denom = lv + rv + 1e-6
                bilateral_asymmetry = abs(lv - rv) / denom
            state["prev_left_hip_y"]  = lh_y_global
            state["prev_right_hip_y"] = rh_y_global

        # ── Post-impact stillness ──────────────────────────────
        post_impact_stillness = 0.0
        if pose_result.has_pose:
            # Snapshot a small set of key landmark Y positions
            key_lm_idx = [0, 11, 12, 23, 24, 27, 28]
            current_snapshot = [
                y1 + pose_result.landmarks[i].y * h
                for i in key_lm_idx
                if i < len(pose_result.landmarks)
            ]
            history = state.setdefault("landmark_history", deque(maxlen=6))
            if len(history) > 0:
                prev_snapshot = history[-1]
                if len(prev_snapshot) == len(current_snapshot):
                    motion = sum(
                        abs(c - p) for c, p in zip(current_snapshot, prev_snapshot)
                    ) / (bh * len(current_snapshot))
                    motion_hist = state.setdefault(
                        "motion_energy_history", deque(maxlen=10)
                    )
                    motion_hist.append(motion)
                    avg_motion = sum(motion_hist) / len(motion_hist)
                    post_impact_stillness = self.clamp01(1.0 - avg_motion / 0.02)
            history.append(current_snapshot)

        # ── 3D torso tilt ──────────────────────────────────────
        torso_tilt_3d = self._torso_tilt_3d(pose_result.world_landmarks)

        # ── Floor proximity ────────────────────────────────────
        frame_h_val  = state.get("frame_h", h) or h
        floor_proximity = self.clamp01((y2 / frame_h_val - 0.3) / 0.7)
        # Scales: bottom 70% of frame → 0..1; top 30% → always 0

        return FeatureVector(
            torso_angle=smoothed_angle,
            aspect_ratio=aspect_ratio,
            hip_velocity=hip_velocity,
            upward_hip_velocity=upward_hip_velocity,
            height_drop=height_drop,
            com_shift=com_shift,
            angle_delta=angle_delta,
            persistence=persistence,
            visibility=visibility,
            smoothed_score_hint=score_hint,
            height_collapse=height_collapse,
            width_expansion=width_expansion,
            head_velocity=head_velocity,
            shoulder_hip_separation=shoulder_hip_separation,
            knee_collapse_left=knee_collapse_left,
            knee_collapse_right=knee_collapse_right,
            bilateral_asymmetry=bilateral_asymmetry,
            post_impact_stillness=post_impact_stillness,
            torso_tilt_3d=torso_tilt_3d,
            floor_proximity=floor_proximity,
        )