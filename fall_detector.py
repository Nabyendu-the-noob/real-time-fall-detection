"""
End-to-end fall detection pipeline.
Layer: Orchestration of Detection -> Tracking -> Pose -> Features -> Score -> State Machine -> Alerts -> Logging -> Visualization
"""

import time
from collections import deque
import cv2

from dashboard import Dashboard, draw_person_box
from feature_extractor import FeatureExtractor
from fall_logger import FallLogger
from person_tracker import PersonTracker
from pose_estimator import PoseEstimator
from temporal_state_machine import TemporalStateMachine
from weighted_scoring import WeightedScorer
from yolo_detector import YOLODetector
from alarm_system import AlarmSystem


SEVERITY_RANK = {"NONE": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}


class FallDetector:
    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        pose_model_path: str = "pose_landmarker.task",
        output_path: str = "output_fall_detected_weighted.mp4",
        display: bool = True,
        display_every_n_frames: int = 5,
    ) -> None:
        self.detector = YOLODetector(model_path=yolo_model_path)
        self.pose = PoseEstimator(model_path=pose_model_path)
        self.tracker = PersonTracker()
        self.features = FeatureExtractor()
        self.scorer = WeightedScorer()
        self.state_machine = TemporalStateMachine()
        self.logger = FallLogger()
        self.dashboard = Dashboard()
        self.alarm = AlarmSystem()

        self.output_path = output_path
        self.display = display
        self.display_every_n_frames = display_every_n_frames
        self.fps_history = deque(maxlen=10)

    def _draw_placeholder(self, frame, text: str) -> None:
        cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def process(self, video_path: str | int) -> str:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_src,
            (w_src, h_src),
        )

        frame_idx = 0
        prev_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                now = time.time()
                self.fps_history.append(now - prev_time)
                fps = 1.0 / (sum(self.fps_history) / len(self.fps_history))
                prev_time = now

                any_fall_active = any(s["fall_detected"] for s in self.tracker._states.values())
                
                recently_fell = any(
                    s.get("fall_reset_time") is not None
                    and (now - s["fall_reset_time"]) < 30.0
                    for s in self.tracker._states.values()
                    if not s["fall_detected"]
                )

                conf = 0.25 if any_fall_active else (0.35 if recently_fell else 0.50)
                detected_boxes = self.detector.detect(frame, conf=conf)
                tracked = self.tracker.update(detected_boxes)

                active_fall_count = 0

                for person_id, box, state in tracked:
                    
                    pose_result = self.pose.estimate(frame, box)
                    state["frame_h"] = frame.shape[0]
                    
                    features = self.features.extract(state, box, pose_result)
                    
                    score_result = self.scorer.score_features(features)
                    
                    state["torso_angle"]  = features.torso_angle
                    state["aspect_ratio"] = features.aspect_ratio
                    state["hip_velocity"] = features.hip_velocity
                    state["upward_hip_velocity"]      = features.upward_hip_velocity
                    state["height_drop"]  = features.height_drop
                    state["visibility"]   = features.visibility
                    state["fall_score"]   = score_result.score
                    state["signal_scores"]  = score_result.signal_scores
                    state["head_velocity"]          = features.head_velocity
                    state["shoulder_hip_separation"]= features.shoulder_hip_separation
                    state["knee_collapse"] = max(features.knee_collapse_left, features.knee_collapse_right)
                    state["bilateral_asymmetry"] = features.bilateral_asymmetry
                    state["post_impact_stillness"] = features.post_impact_stillness
                    state["torso_tilt_3d"] = features.torso_tilt_3d
                    state["angle_delta"]   = features.angle_delta
                    state["last_pose"] = pose_result
                    
                    # ── 5. Update score history
                    self.features.update_score_history(state, score_result.score)

                    # ── 6. Severity — escalate only, never downgrade
                    if state["fall_detected"]:
                        prev = state.get("severity", "NONE")
                        if SEVERITY_RANK[score_result.severity] > SEVERITY_RANK[prev]:
                            state["severity"] = score_result.severity
                    else:
                        state["severity"] = score_result.severity
                    
                    prev_fall_detected = state.get("fall_detected", False)
                    
                    # ── 7. State machine
                    transition = self.state_machine.update(state, score_result, now_t=now)


                    # ── 8. Alarm + dashboard on confirmed fall
                    if transition.confirmed_fall:
                        state["fall_start_time"] = now
                        self.dashboard.update_fall(score_result.severity)
                        self.alarm.trigger()

                    # ── 9. Duration tracking + CSV logging
                    if state["fall_detected"] and state.get("fall_start_time") is not None:
                        state["fall_duration"] = time.time() - state["fall_start_time"]
                        if transition.confirmed_fall:
                            self.logger.log(
                                frame,
                                person_id,
                                score_result.severity,
                                state["fall_duration"],
                                score=score_result.score,
                            )

                    if (state["fall_detected"]
                            and state.get("fall_duration", 0) >= 5.0):
                        self.alarm.escalate()       # continuous 1400 Hz loop
 
                    # ── 11. Alarm — reset when person gets up ──────────────
                    if prev_fall_detected and not state["fall_detected"]:
                        self.alarm.reset()
                    
                    if state["fall_detected"]:
                        active_fall_count += 1

                    # ── 10. Visualise
                    draw_person_box(frame, box, person_id, state)

                # ── Dashboard fall-active banner logic
                if active_fall_count > 0:
                    self.dashboard.fall_active = True
                    self.dashboard.fall_lost_frames = 0
                elif len(tracked) == 0 and self.dashboard.fall_active:
                    self.dashboard.fall_lost_frames += 1
                    if self.dashboard.fall_lost_frames > self.dashboard.FALL_LOST_TOLERANCE:
                        self.dashboard.fall_active = False
                        self.dashboard.fall_lost_frames = 0
                elif len(tracked) > 0 and active_fall_count == 0:
                    self.dashboard.fall_active = False
                    self.dashboard.fall_lost_frames = 0

                frame = self.dashboard.draw(frame, len(tracked), fps)
                out.write(frame)
                
                # ── Live preview
                if self.display and frame_idx % self.display_every_n_frames == 0:
                    cv2.imshow("Fall Detection", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                # if frame_idx % max(1, self.display_every_n_frames) == 0:
                #     print(
                #         f"[Frame {frame_idx}/{total_frames}] FPS: {fps:.1f} | "
                #         f"Persons: {len(tracked)} | Active falls: {active_fall_count} | "
                #         f"Total falls: {self.dashboard.total_falls}"
                #     )

        finally:
            cap.release()
            out.release()
            self.pose.close()
            self.alarm.close()
            cv2.destroyAllWindows()

        # print("\n" + "=" * 55)
        # print("Processing complete.")
        # print(f"Output video : {self.output_path}")
        # print("Fall log CSV : fall_events.csv")
        # print("Screenshots  : fall_logs/")
        # print(f"Total falls  : {self.dashboard.total_falls}")
        # print("=" * 55)
        
        return self.output_path
    
