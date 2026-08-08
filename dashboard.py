import time
from collections import deque

import cv2

STATE_COLORS = {
    "NORMAL":    (0, 220, 0),
    "CANDIDATE": (0, 165, 255),
    "FALLEN":    (0, 0, 255),
}

SEVERITY_COLORS = {
    "NONE":     (0, 220, 0),
    "MILD":     (0, 220, 220),
    "MODERATE": (0, 140, 255),
    "SEVERE":   (0, 0, 255),
}


# MediaPipe pose connections (landmark index pairs)
POSE_CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso sides
    (23, 24),            # hips
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
]

# Key landmarks to draw dots on
KEY_LANDMARKS = {
    11: "L.Sh", 12: "R.Sh",
    13: "L.El", 14: "R.El",
    23: "L.Hip", 24: "R.Hip",
    25: "L.Kn",  26: "R.Kn",
    27: "L.Ank", 28: "R.Ank",
}


class Dashboard:
    """Renders the HUD overlay onto the frame (stats panel + fall banner)."""

    def __init__(self):
        self.total_falls         = 0
        self.last_fall_time      = "None"
        self.fall_active         = False
        self.fall_lost_frames    = 0
        self.FALL_LOST_TOLERANCE = 30
        self.frame_times         = deque(maxlen=10)

    def update_fall(self, severity: str) -> None:
        self.total_falls   += 1
        self.last_fall_time = time.strftime("%H:%M:%S")
        self.fall_active    = True

    def draw(self, frame, active_persons: int, fps: float):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (295, 145), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

        rows = [
            (f"FPS: {fps:.1f}",                   (0, 220, 0)),
            (f"Persons: {active_persons}",         (220, 220, 0)),
            (f"Total Falls: {self.total_falls}",   (50, 80, 255)),
            (f"Last Fall: {self.last_fall_time}",  (180, 180, 180)),
        ]
        for i, (txt, col) in enumerate(rows):
            cv2.putText(frame, txt, (18, 32 + i * 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

        if self.fall_active and active_persons == 0:
            cv2.putText(
                frame, "! FALL ACTIVE — PERSON ON GROUND !",
                (frame.shape[1] // 2 - 240, frame.shape[0] - 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 255), 2,
            )
        return frame


def _draw_skeleton(frame, box: list, pose_result) -> None:
    """Draw MediaPipe keypoints and skeleton connections onto the frame."""
    if pose_result is None or not pose_result.has_pose:
        return

    x1, y1, x2, y2 = box
    h, w = pose_result.crop_shape[:2]
    landmarks = pose_result.landmarks

    def to_frame(lm):
        px = int(x1 + lm.x * w)
        py = int(y1 + lm.y * h)
        return px, py

    # Draw skeleton connections
    for a, b in POSE_CONNECTIONS:
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        vis_a = getattr(landmarks[a], "visibility", 0)
        vis_b = getattr(landmarks[b], "visibility", 0)
        if vis_a < 0.3 or vis_b < 0.3:
            continue
        pt_a = to_frame(landmarks[a])
        pt_b = to_frame(landmarks[b])
        cv2.line(frame, pt_a, pt_b, (0, 200, 255), 1, cv2.LINE_AA)

    # Draw landmark dots
    for idx, label in KEY_LANDMARKS.items():
        if idx >= len(landmarks):
            continue
        lm  = landmarks[idx]
        vis = getattr(lm, "visibility", 0)
        if vis < 0.3:
            continue
        pt = to_frame(lm)
        # Hips in yellow (most important for fall), others in cyan
        dot_color = (0, 255, 255) if idx in (23, 24) else (255, 255, 0)
        cv2.circle(frame, pt, 4, dot_color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 4, (0, 0, 0),  1, cv2.LINE_AA)


def _draw_factor_panel(frame, box: list, state: dict) -> None:
    """
    Draw a colour-coded factor panel beside the bounding box showing
    all key fall signals with bar indicators and raw numeric values.
    """
    x1, y1, x2, y2 = box
    sig   = state.get("signal_scores", {})
    score = state.get("fall_score", 0.0)
    angle = state.get("torso_angle")
    ar    = state.get("aspect_ratio", 0.0)
    vel   = state.get("hip_velocity", 0.0)
    drop  = state.get("height_drop", 0.0)
    vis   = state.get("visibility", 0.0)

    factors = [
        ("Score",   score,                        (200, 200, 200)),
        ("Angle",   sig.get("angle",      0),     (100, 200, 255)),
        ("A/R",     sig.get("ratio",      0),     (100, 255, 200)),
        ("Drop",    sig.get("height_drop",0),     (255, 200, 100)),
        ("Vel",     sig.get("velocity",   0),     (255, 100, 200)),
        ("HeadV",   sig.get("head_vel",   0),     (255, 180,  80)),  
        ("Sep",     sig.get("separation", 0),     (180, 255, 100)),  
        ("Tilt3D",  sig.get("tilt_3d",   0),     (100, 180, 255)),  
        ("Knee",    sig.get("knee",       0),     (255, 120, 120)),  
        ("Still",   sig.get("stillness",  0),     (120, 255, 180)),  
        ("Persist", sig.get("persistence",0),     (200, 100, 255)),
        ("Vis",     vis,                          (180, 180, 180)),
    ]

    bar_max = 70
    panel_w = bar_max + 58
    row_h   = 16
    panel_h = len(factors) * row_h + 6

    # Try to place panel to the right of the box
    panel_x = x2 + 8
    panel_y = y1
    fits_right = (panel_x + panel_w < frame.shape[1]) and (panel_y + panel_h < frame.shape[0])

    if not fits_right:
        # Fall back: draw below the box
        panel_x = x1
        panel_y = y2 + 4
        if panel_y + panel_h >= frame.shape[0]:
            return   # no room at all — skip

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x - 2, panel_y - 2),
                  (panel_x + panel_w, panel_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, (label, val, color) in enumerate(factors):
        ry = panel_y + 4 + i * row_h

        cv2.putText(frame, f"{label}:", (panel_x + 2, ry + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        bx = panel_x + 46
        cv2.rectangle(frame, (bx, ry + 2), (bx + bar_max, ry + 12),
                      (50, 50, 50), -1)

        fill  = max(0.0, min(1.0, float(val)))
        bar_w = int(fill * bar_max)
        if bar_w > 0:
            r = int(255 * fill)
            g = int(255 * (1 - fill))
            cv2.rectangle(frame, (bx, ry + 2), (bx + bar_w, ry + 12),
                          (0, g, r), -1)

        cv2.putText(frame, f"{val:.2f}", (bx + bar_max + 2, ry + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

    # Raw values row beneath panel
    raw_y = panel_y + panel_h + 2
    if angle is not None and raw_y + 12 < frame.shape[0]:
        cv2.putText(
            frame,
            f"ang={angle:.0f}  ar={ar:.2f}  vel={vel:.2f}  drop={drop:.2f}",
            (panel_x, raw_y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1,
        )


def draw_person_box(frame, box: list, pid: int, state: dict) -> None:
    """
    Draws bounding box, FSM state label, MediaPipe skeleton,
    factor panel, mini signal bars, and fall banner.
    """
    x1, y1, x2, y2 = box
    fs_value = "FALLEN" if state.get("fall_detected") else (
           "CANDIDATE" if state.get("candidate_frames", 0) > 0 else "NORMAL")
    sev   = state["severity"]
    score = state["fall_score"]
    color = STATE_COLORS.get(fs_value, (0, 200, 0))

    # ── Bounding box ──────────────────────────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # ── State + score label ───────────────────────────────────────────────────
    label = f"P{pid}  [{fs_value}]  score={score:.2f}"
    if state["fall_detected"]:
        label += f"  {sev}  {state['fall_duration']:.1f}s"
    if state.get("torso_angle") is not None:
        label += f"  ang={state['torso_angle']:.0f}deg"
    cv2.putText(frame, label, (x1, max(y1 - 10, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    # ── MediaPipe skeleton overlay ────────────────────────────────────────────
    _draw_skeleton(frame, box, state.get("last_pose"))

    # ── Key factor panel ──────────────────────────────────────────────────────
    _draw_factor_panel(frame, box, state)

    # ── Mini signal score bars ────────────────────────────────────────────────
    # sig = state.get("signal_scores", {})
    # bar_labels = [
    #     ("ang", sig.get("angle",      0)),
    #     ("hgt", sig.get("height_drop",0)),
    #     ("vel", sig.get("velocity",   0)),
    #     ("rat", sig.get("ratio",      0)),
    #     ("hdv", sig.get("head_vel",   0)),   # new
    #     ("stp", sig.get("stillness",  0)),   # new
    #     ("3dt", sig.get("tilt_3d",   0)),   # new
    #     ("per", sig.get("persistence",0)),
    # ]
    # bx = x1
    # for name, val in bar_labels:
    #     bar_w    = int(val * 22)
    #     fill_col = (0, int(200 * (1 - val)), int(200 * val))
    #     cv2.rectangle(frame, (bx, y1 - 26), (bx + 22, y1 - 18), (40, 40, 40), -1)
    #     if bar_w > 0:
    #         cv2.rectangle(frame, (bx, y1 - 26), (bx + bar_w, y1 - 18), fill_col, -1)
    #     cv2.putText(frame, name, (bx + 1, y1 - 18),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)
    #     bx += 26

    # ── Fall banner ───────────────────────────────────────────────────────────
    if state["fall_detected"]:
        sev_col = SEVERITY_COLORS.get(sev, (0, 0, 255))
        cv2.putText(
            frame, f"  ! FALL: {sev} !",
            (frame.shape[1] // 2 - 160, frame.shape[0] - 18),
            cv2.FONT_HERSHEY_DUPLEX, 0.9, sev_col, 2,
        )