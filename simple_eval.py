"""
simple_eval.py  —  Evaluate Fall 2.0 on URFD using main.py as a subprocess.
Place in Fall2.0/ folder alongside main.py.

Usage:
    python simple_eval.py --dataset C:/datasets/URFD
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
NUM_FALL = 30
NUM_ADL  = 40
LOG_CSV  = "fall_events.csv"
TEMP_OUT = "__temp_eval_output.mp4"


def clean():
    """Delete log file and screenshots before each video."""
    if os.path.exists(LOG_CSV):
        os.remove(LOG_CSV)
    if os.path.exists("fall_logs"):
        shutil.rmtree("fall_logs")
    if os.path.exists(TEMP_OUT):
        os.remove(TEMP_OUT)


def fall_was_detected() -> bool:
    """Return True if fall_events.csv has at least one data row."""
    if not os.path.exists(LOG_CSV):
        return False
    with open(LOG_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return len(rows) > 1   # row 0 is header


def run_video(video_path: str) -> bool:
    """
    Run main.py on one video in a fresh subprocess.
    Returns True if a fall was detected.
    """
    clean()

    result = subprocess.run(
        [
            sys.executable, "main.py",
            "--source",  video_path,
            "--output",  TEMP_OUT,
        ],
        capture_output=True,   # suppress console output during batch run
        text=True,
    )

    if result.returncode != 0:
        print(f"\n  [ERROR] {result.stderr.strip()[:120]}")

    return fall_was_detected()


def compute_metrics(tp, fn, fp, tn):
    total = tp + fn + fp + tn
    sens  = tp / (tp + fn)          if (tp + fn) > 0 else 0
    spec  = tn / (tn + fp)          if (tn + fp) > 0 else 0
    prec  = tp / (tp + fp)          if (tp + fp) > 0 else 0
    f1    = 2*prec*sens/(prec+sens) if (prec + sens) > 0 else 0
    acc   = (tp + tn) / total       if total > 0 else 0
    far   = fp / (fp + tn)          if (fp + tn) > 0 else 0

    import math
    den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn - fp*fn) / den     if den > 0 else 0

    return dict(
        TP=tp, FN=fn, FP=fp, TN=tn,
        Sensitivity=round(sens*100, 2),
        Specificity=round(spec*100, 2),
        Precision  =round(prec*100, 2),
        F1         =round(f1  *100, 2),
        Accuracy   =round(acc *100, 2),
        FAR        =round(far *100, 2),
        MCC        =round(mcc,      4),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Path to URFD folder")
    parser.add_argument("--cam", default="cam0",
                        choices=["cam0", "cam1"])
    parser.add_argument("--output-csv", default="eval_results.csv")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    cam     = args.cam

    # Build video list: (path, label)
    videos = []
    for i in range(1, NUM_FALL + 1):
        p = dataset / f"fall-{i:02d}-{cam}-rgb.avi"
        if p.exists():
            videos.append((str(p), "FALL"))
        else:
            print(f"  [skip] not found: {p.name}")

    for i in range(1, NUM_ADL + 1):
        p = dataset / f"adl-{i:02d}-{cam}-rgb.avi"
        if p.exists():
            videos.append((str(p), "NO_FALL"))
        else:
            print(f"  [skip] not found: {p.name}")

    print(f"\nFound {len(videos)} videos — starting evaluation\n")

    rows    = []
    tp = fn = fp = tn = 0

    for idx, (vpath, label) in enumerate(videos, 1):
        name = Path(vpath).name
        print(f"[{idx:>2}/{len(videos)}] {name:<38} GT={label:<8}", end="", flush=True)

        detected  = run_video(vpath)
        predicted = "FALL" if detected else "NO_FALL"

        if   label == "FALL"    and detected:     tp += 1; outcome = "TP"; mark = "✓"
        elif label == "FALL"    and not detected:  fn += 1; outcome = "FN"; mark = "✗ MISSED"
        elif label == "NO_FALL" and detected:      fp += 1; outcome = "FP"; mark = "✗ FALSE ALARM"
        else:                                      tn += 1; outcome = "TN"; mark = "✓"

        print(f" → {outcome}  {mark}")
        rows.append({"video": name, "label": label,
                     "predicted": predicted, "outcome": outcome})

    # ── Print summary ─────────────────────────────────────────────────────────
    m = compute_metrics(tp, fn, fp, tn)
    print()
    print("=" * 52)
    print("  RESULTS — UR Fall Detection Dataset")
    print("=" * 52)
    print(f"  TP={m['TP']}  FN={m['FN']}  FP={m['FP']}  TN={m['TN']}")
    print()
    print(f"  Sensitivity  : {m['Sensitivity']:>6.2f}%  ← most important")
    print(f"  Specificity  : {m['Specificity']:>6.2f}%")
    print(f"  Precision    : {m['Precision']:>6.2f}%")
    print(f"  F1 Score     : {m['F1']:>6.2f}%")
    print(f"  Accuracy     : {m['Accuracy']:>6.2f}%")
    print(f"  False Alarm  : {m['FAR']:>6.2f}%")
    print(f"  MCC          : {m['MCC']:>7.4f}")
    print("=" * 52)

    # Missed falls
    missed = [r["video"] for r in rows if r["outcome"] == "FN"]
    if missed:
        print(f"\n  Missed falls ({len(missed)}):")
        for v in missed: print(f"    - {v}")

    false_alarms = [r["video"] for r in rows if r["outcome"] == "FP"]
    if false_alarms:
        print(f"\n  False alarms ({len(false_alarms)}):")
        for v in false_alarms: print(f"    - {v}")

    # ── Save per-video CSV ────────────────────────────────────────────────────
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video","label","predicted","outcome"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Per-video results saved → {args.output_csv}")

    clean()


if __name__ == "__main__":
    main()