from dataclasses import dataclass


@dataclass
class ScoreResult:
    score: float
    severity: str
    candidate: bool
    signal_scores: dict


class WeightedScorer:
    def __init__(self):
        self.threshold = 0.30

    @staticmethod
    def clamp01(v):
        return max(0.0, min(1.0, float(v)))

    def score_features(self, f):
        angle_score = self.clamp01(f.torso_angle / 90)
        velocity_score = self.clamp01(f.hip_velocity / 0.35)
        drop_score = self.clamp01((f.height_drop - 0.05) / 0.50)
        ratio_score = self.clamp01((f.aspect_ratio - 0.55) / (1.2 - 0.55))
        persistence_score = f.persistence
        collapse_score = self.clamp01(f.height_collapse / 0.5)
        width_score = self.clamp01(f.width_expansion / 1.0)
        
        head_vel_score  = self.clamp01(f.head_velocity / 0.40)
        sep_score       = self.clamp01(1.0 - f.shoulder_hip_separation / 0.5)
        # sep=0.5 (normal) → 0; sep=0 (collapsed) → 1
        knee_score      = max(f.knee_collapse_left, f.knee_collapse_right)
        asym_score      = self.clamp01(f.bilateral_asymmetry / 0.6)
        # stillness_score = f.post_impact_stillness
        tilt_3d_score   = self.clamp01(f.torso_tilt_3d / 75)
        ang_vel_score   = self.clamp01(f.angle_delta / 25)
        
        if f.aspect_ratio < 0.8:
            stillness_score = 0.0
        else:
            stillness_score = f.post_impact_stillness
            
        if hasattr(f, "upward_hip_velocity") and f.upward_hip_velocity > 0.05:
            angle_score *= 0.20

        score = (
            angle_score       * 0.04
            + velocity_score  * 0.16
            + drop_score      * 0.14
            + ratio_score     * 0.15
            + persistence_score * 0.04
            + collapse_score  * 0.10
            + width_score     * 0.06
            + head_vel_score  * 0.14   
            + sep_score       * 0.02   # new — shoulder/hip separation
            + knee_score      * 0.02   # new — leg collapse
            + asym_score      * 0.04   # new — bilateral
            + stillness_score * 0.03   # new — post-impact
            + tilt_3d_score   * 0.02   # new — 3D tilt
            + ang_vel_score   * 0.04   # new — angular velocity
        )
        
        if (f.aspect_ratio > 1.6 and f.hip_velocity > 0.10):
            score += 0.09
        
        if f.aspect_ratio > 1.2 and f.height_collapse > 0.20:
            score += 0.09
        
        if f.head_velocity > 0.20 and f.torso_tilt_3d > 40 and f.aspect_ratio > 0.55:
            score += 0.09
            
        if stillness_score > 0.7 and ratio_score > 0.5 and velocity_score > 0.2:
            score += 0.09
        
        if f.height_drop > 0.60 and f.torso_tilt_3d > 40 and f.aspect_ratio > 0.55:
            score += 0.10
        
        if f.shoulder_hip_separation > 0.20 and f.torso_tilt_3d > 40 and f.aspect_ratio > 0.55:
            score += 0.08
        
        if f.height_drop >= 0.95 and f.aspect_ratio > 0.55:
            score += 0.07
            
        score = self.clamp01(score)
        
        if hasattr(f, "upward_hip_velocity"):
            recovery_suppression = self.clamp01(f.upward_hip_velocity / 0.15)
            score *= (1.0 - recovery_suppression * 0.60)

        reliability = self.clamp01((f.visibility - 0.4) / 0.6)
        score *= (0.8 + reliability * 0.2)
        
        score = self.clamp01(score)
        
        severity_score = (
            velocity_score * 0.35
            + collapse_score * 0.30
            + drop_score * 0.30
        )

        severity = "NONE"

        if severity_score > 0.75:
            severity = "SEVERE"

        elif severity_score > 0.50:
            severity = "MODERATE"

        elif score > self.threshold:
            severity = "MILD"

        # print(
        #     f"A={angle_score:.2f}",
        #     f"V={velocity_score:.2f}",
        #     f"D={drop_score:.2f}",
        #     f"R={ratio_score:.2f}",
        #     f"P={persistence_score:.2f}",
        #     f"upV={getattr(f, 'upward_hip_velocity', 0):.3f}",
        #     f"FINAL={score:.2f}",
        # )
        
        return ScoreResult(
            score=score,
            severity=severity,
            candidate=score > self.threshold,
            signal_scores={
                "angle":       angle_score,
                "velocity":    velocity_score,
                "height_drop": drop_score,
                "ratio":       ratio_score,
                "persistence": persistence_score,
                "collapse":    collapse_score,
                "width":       width_score,
                "head_vel":     head_vel_score,
                "separation":   sep_score,
                "knee":         knee_score,
                "asymmetry":    asym_score,
                "stillness":    stillness_score,
                "tilt_3d":      tilt_3d_score,
                "ang_vel":      ang_vel_score,
            },
        )