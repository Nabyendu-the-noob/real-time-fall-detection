# Weighted Fall Detection Refactor

## Files
- `main.py` — CLI entrypoint
- `fall_detector.py` — end-to-end pipeline
- `yolo_detector.py` — detection
- `person_tracker.py` — tracking
- `pose_estimator.py` — pose inference
- `feature_extractor.py` — feature engineering
- `weighted_scoring.py` — weighted score + severity
- `temporal_state_machine.py` — confirmation/reset logic
- `fall_logger.py` — CSV + screenshots
- `dashboard.py` — visualization overlay

## Setup
1. Create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download these files into the same folder:
   - `yolov8n.pt`
   - `pose_landmarker.task`

## Run
```bash
python main.py --video path/to/video.mp4
```

To disable the preview window:
```bash
python main.py --video path/to/video.mp4 --no-display
```

## Output
- `output_fall_detected_weighted.mp4`
- `fall_events.csv`
- `fall_logs/`
