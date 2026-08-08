"""
Temporal smoothing, confirmation and reset logic.
Layer: State Machine
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StateTransition:
    confirmed_fall: bool
    reset_fall: bool
    active_fall: bool
    event_type: str | None = None


class TemporalStateMachine:
    def __init__(self):
        self.confirm_frames = 8
        self.reset_frames = 20

    def update(self, state, score_result, now_t=None):
        state.setdefault("candidate_frames", 0)
        state.setdefault("normal_frames", 0)
        state.setdefault("fall_detected", False)
        state.setdefault("fall_confirmed_time", None)   # when fall first confirmed
        state.setdefault("fall_reset_time", None)       # when fall last cleared

        if score_result.candidate:
            state["candidate_frames"] += 1
            state["normal_frames"] = 0
        else:
            state["normal_frames"] += 1
            state["candidate_frames"] = 0

        confirmed = False
        
        if score_result.score > 0.55:
            if not state["fall_detected"]:
                confirmed = True
                state["fall_confirmed_time"] = now_t 
            # confirmed = not state["fall_detected"] # only confirm if not already detected

            state["fall_detected"] = True
            state["candidate_frames"] = self.confirm_frames

            return StateTransition(
                confirmed_fall=confirmed,
                reset_fall=False,
                active_fall=True
            )

        if state["candidate_frames"] >= self.confirm_frames:
            if not state["fall_detected"]:
                confirmed = True
                state["fall_confirmed_time"] = now_t   # stamp the confirmation moment
            state["fall_detected"] = True
            state["candidate_frames"] = self.confirm_frames  # cap it

        standard_reset = state["normal_frames"] >= self.reset_frames
        
        ar    = state.get("aspect_ratio", 1.0)
        angle = state.get("torso_angle", 90.0)
        
        clearly_upright = (ar < 0.35 and angle < 18.0)
        geometry_reset  = clearly_upright and state["normal_frames"] >= 5

        if standard_reset or geometry_reset:
            state["fall_detected"]   = False
            state["candidate_frames"] = 0
            state["fall_reset_time"] = now_t
        
        # if state["normal_frames"] >= self.reset_frames:
        #     state["fall_detected"] = False
        #     state["candidate_frames"] = 0
        #     state["fall_reset_time"] = now_t           # stamp the reset moment

        return StateTransition(
            confirmed_fall=confirmed,
            reset_fall=state["normal_frames"] >= self.reset_frames,
            active_fall=state["fall_detected"],
    )