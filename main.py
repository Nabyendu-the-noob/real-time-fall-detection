"""
Local machine entrypoint for VS Code / terminal execution.

Examples:
    python main.py
    python main.py --source fall_demo.mp4
    python main.py --source 0
"""


import argparse
import os


from fall_detector import FallDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Weighted fall detection pipeline"
    )

    parser.add_argument(
        "--source",
        default=r"fall01.mp4",
        help="Video path OR webcam index"
    )

    parser.add_argument(
        "--output",
        default="output_fall_detected_weighted.mp4",
        help="Path to output video"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV preview window"
    )

    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="YOLO model path"
    )

    parser.add_argument(
        "--pose-model",
        default="pose_landmarker.task",
        help="MediaPipe pose task path"
    )

    parser.add_argument(
        "--display-every",
        type=int,
        default=5,
        help="Show preview every N frames"
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Convert webcam string -> integer
    video_source = (
        int(args.source)
        if str(args.source).isdigit()
        else args.source
    )

    # Validate video path only if NOT webcam
    if isinstance(video_source, str):
        if not os.path.exists(video_source):
            print(f"[ERROR] Video not found: {video_source}")
            return 1

    # Validate pose model
    if not os.path.exists(args.pose_model):
        print(f"[ERROR] Pose model not found: {args.pose_model}")
        print("Download pose_landmarker.task and place it beside these files.")
        return 1

    detector = FallDetector(
        yolo_model_path=args.yolo_model,
        pose_model_path=args.pose_model,
        output_path=args.output,
        display=not args.no_display,
        display_every_n_frames=args.display_every,
    )

    try:
        detector.process(video_source)
        return 0

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        return 130

    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    main()