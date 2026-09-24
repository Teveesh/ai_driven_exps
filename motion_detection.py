"""
Motion Detection & Analysis for Video Files
============================================
This script analyzes a video file and detects the level of motion
in each frame using frame differencing techniques.

It computes:
  - Per-frame motion score (mean absolute difference between frames)
  - Percentage of changed pixels per frame
  - Motion level classification (Low / Medium / High)
  - Overall video motion statistics
  - A timeline of motion levels over the video duration
  - Optional: saves a motion report (CSV) and a motion heatmap image

Usage:
    python motion_detection.py "path/to/video.mp4"
    python motion_detection.py "path/to/video.mp4" --threshold 25 --report
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Motion level classification thresholds
# ----------------------------------------------------------------------
# These thresholds define the boundaries between motion levels based on
# the mean absolute difference (MAD) of pixel intensities between
# consecutive frames (0-255 scale).
LOW_MOTION_THRESHOLD = 5.0      # MAD below this = Low motion
MEDIUM_MOTION_THRESHOLD = 15.0  # MAD below this = Medium motion, else High

# Percentage of changed pixels threshold (0-100)
CHANGED_PIXEL_THRESHOLD = 10.0  # % of pixels that changed significantly


def classify_motion(mad_score: float, changed_pct: float) -> str:
    """
    Classify the motion level based on the mean absolute difference
    and the percentage of changed pixels.

    Args:
        mad_score: Mean absolute difference between consecutive frames (0-255)
        changed_pct: Percentage of pixels that changed significantly (0-100)

    Returns:
        One of: "Low", "Medium", "High"
    """
    if mad_score < LOW_MOTION_THRESHOLD and changed_pct < CHANGED_PIXEL_THRESHOLD:
        return "Low"
    elif mad_score < MEDIUM_MOTION_THRESHOLD:
        return "Medium"
    else:
        return "High"


def analyze_motion(
    video_path: str,
    pixel_threshold: int = 25,
    report: bool = False,
    report_dir: str = "motion_reports",
    progress_interval: int = 100,
) -> None:
    """
    Analyze motion levels in a video file.

    Args:
        video_path: Path to the video file
        pixel_threshold: Pixel intensity difference threshold for
                         considering a pixel "changed" (0-255)
        report: If True, save a CSV report and a motion heatmap image
        report_dir: Directory to save reports
        progress_interval: Print progress every N frames
    """
    video_path = video_path.strip().strip('"').strip("'")
    path = Path(video_path).expanduser()

    if not path.is_file():
        print(f"Error: Video file not found: {path}")
        return

    # Open the video
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        print("Error: Could not open video file.")
        return

    # Video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0  # fallback

    duration = total_frames / fps if total_frames > 0 else 0

    print("=" * 60)
    print("MOTION DETECTION & ANALYSIS")
    print("=" * 60)
    print(f"Video File      : {path.name}")
    print(f"Resolution      : {frame_width} x {frame_height}")
    print(f"Total Frames    : {total_frames}")
    print(f"Frame Rate      : {fps:.2f} FPS")
    print(f"Duration        : {duration:.2f} seconds")
    print(f"Pixel Threshold : {pixel_threshold}")
    print("=" * 60)
    print()

    # Storage for per-frame results
    frame_numbers = []
    timestamps = []
    mad_scores = []
    changed_pcts = []
    motion_levels = []

    # Read the first frame
    success, prev_frame = video.read()
    if not success:
        print("Error: Could not read the first frame.")
        video.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    frame_number = 1
    total_mad = 0.0
    total_changed = 0.0
    level_counts = {"Low": 0, "Medium": 0, "High": 0}

    print("Analyzing frames...")

    while True:
        success, frame = video.read()
        if not success:
            break

        # Convert to grayscale and blur to reduce noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Compute absolute difference between current and previous frame
        diff = cv2.absdiff(prev_gray, gray)

        # Mean absolute difference (motion score)
        mad = float(np.mean(diff))

        # Percentage of changed pixels
        changed_mask = diff > pixel_threshold
        changed_pct = float(np.count_nonzero(changed_mask) / diff.size * 100.0)

        # Classify motion level
        level = classify_motion(mad, changed_pct)

        # Record results
        frame_numbers.append(frame_number)
        timestamps.append(frame_number / fps)
        mad_scores.append(mad)
        changed_pcts.append(changed_pct)
        motion_levels.append(level)

        # Accumulate totals
        total_mad += mad
        total_changed += changed_pct
        level_counts[level] += 1

        # Progress indicator
        if frame_number % progress_interval == 0:
            print(f"  Processed {frame_number}/{total_frames} frames "
                  f"({frame_number / total_frames * 100:.1f}%)")

        # Update previous frame
        prev_gray = gray
        frame_number += 1

    video.release()

    if frame_number <= 1:
        print("Error: No frames were processed.")
        return

    # ------------------------------------------------------------------
    # Compute overall statistics
    # ------------------------------------------------------------------
    analyzed_frames = frame_number - 1  # exclude the first frame (no diff)
    avg_mad = total_mad / analyzed_frames if analyzed_frames > 0 else 0
    avg_changed = total_changed / analyzed_frames if analyzed_frames > 0 else 0

    # Determine overall motion level
    if avg_mad < LOW_MOTION_THRESHOLD and avg_changed < CHANGED_PIXEL_THRESHOLD:
        overall_level = "Low"
    elif avg_mad < MEDIUM_MOTION_THRESHOLD:
        overall_level = "Medium"
    else:
        overall_level = "High"

    # Find peak motion frame
    if mad_scores:
        peak_idx = int(np.argmax(mad_scores))
        peak_frame = frame_numbers[peak_idx]
        peak_time = timestamps[peak_idx]
        peak_mad = mad_scores[peak_idx]
    else:
        peak_frame = 0
        peak_time = 0
        peak_mad = 0

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("MOTION ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Frames Analyzed     : {analyzed_frames}")
    print(f"Average Motion Score: {avg_mad:.2f} (0-255 scale)")
    print(f"Avg Changed Pixels  : {avg_changed:.2f}%")
    print(f"Overall Motion Level: {overall_level}")
    print()
    print("Motion Level Distribution:")
    for level in ["Low", "Medium", "High"]:
        count = level_counts[level]
        pct = count / analyzed_frames * 100 if analyzed_frames > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {level:8s}: {count:6d} frames ({pct:5.1f}%) {bar}")
    print()
    print(f"Peak Motion Frame   : {peak_frame} (at {peak_time:.2f}s)")
    print(f"Peak Motion Score   : {peak_mad:.2f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Motion timeline (segments of 10% of the video)
    # ------------------------------------------------------------------
    print()
    print("MOTION TIMELINE (10% segments):")
    num_segments = 10
    segment_size = max(1, analyzed_frames // num_segments)
    for seg in range(num_segments):
        start = seg * segment_size
        end = min((seg + 1) * segment_size, analyzed_frames)
        if start >= end:
            break
        seg_mads = mad_scores[start:end]
        seg_avg = float(np.mean(seg_mads)) if seg_mads else 0
        seg_level = classify_motion(seg_avg, avg_changed)
        seg_start_time = start / fps
        seg_end_time = (end - 1) / fps
        bar_len = int(seg_avg / 2)
        bar = "#" * min(bar_len, 50)
        print(f"  [{seg_start_time:6.1f}s-{seg_end_time:6.1f}s] "
              f"score={seg_avg:5.1f}  {seg_level:6s}  {bar}")

    # ------------------------------------------------------------------
    # Save report if requested
    # ------------------------------------------------------------------
    if report:
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)

        # CSV report
        csv_file = report_path / f"{path.stem}_motion_report.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame", "timestamp_s", "motion_score", "changed_pct", "motion_level"
            ])
            for i in range(analyzed_frames):
                writer.writerow([
                    frame_numbers[i],
                    f"{timestamps[i]:.3f}",
                    f"{mad_scores[i]:.3f}",
                    f"{changed_pcts[i]:.3f}",
                    motion_levels[i],
                ])
        print(f"\nCSV report saved: {csv_file}")

        # Motion heatmap image (motion score over time)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(timestamps, mad_scores, color="tab:blue", linewidth=0.8)
            ax.axhline(LOW_MOTION_THRESHOLD, color="green", linestyle="--",
                       label=f"Low threshold ({LOW_MOTION_THRESHOLD})")
            ax.axhline(MEDIUM_MOTION_THRESHOLD, color="orange", linestyle="--",
                       label=f"Medium threshold ({MEDIUM_MOTION_THRESHOLD})")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Motion Score (MAD)")
            ax.set_title(f"Motion Analysis - {path.name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            heatmap_file = report_path / f"{path.stem}_motion_heatmap.png"
            fig.savefig(heatmap_file, dpi=150)
            plt.close(fig)
            print(f"Heatmap saved: {heatmap_file}")
        except ImportError:
            print("Note: matplotlib not available - skipping heatmap generation.")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect and analyze motion levels in a video file."
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=r"C:\Users\tevee\Downloads\From Klickpin.com- 89 Trending Summer Salad Recipes for This Year-pin-id-296674694227142129.mp4",
        help="Path to the video file",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=25,
        help="Pixel intensity threshold for changed pixels (default: 25)",
    )
    parser.add_argument(
        "--report",
        "-r",
        action="store_true",
        help="Save a CSV report and motion heatmap",
    )
    parser.add_argument(
        "--report-dir",
        default="motion_reports",
        help="Directory to save reports (default: motion_reports)",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=100,
        help="Print progress every N frames (default: 100)",
    )

    args = parser.parse_args()

    analyze_motion(
        video_path=args.video_path,
        pixel_threshold=args.threshold,
        report=args.report,
        report_dir=args.report_dir,
        progress_interval=args.progress,
    )