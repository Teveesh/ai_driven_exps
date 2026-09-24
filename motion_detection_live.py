"""
Real-Time Motion Detection with Direction Arrows
=================================================
This script plays a video in a window while performing real-time
motion detection using optical flow. It shows:

  - The video playing with GREEN ARROWS showing motion direction
  - A live motion score bar (0-100)
  - Current motion level (Low / Medium / High)
  - Frame counter and FPS

Controls:
  Q or Esc : Quit
  Space    : Pause / Resume
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def run_live_motion_detection(video_path: str, pixel_threshold: int = 25) -> None:
    """
    Play a video with real-time motion detection and direction arrows.

    Args:
        video_path: Path to the video file
        pixel_threshold: Pixel intensity difference threshold for motion
    """
    video_path = video_path.strip().strip('"').strip("'")
    path = Path(video_path).expanduser()

    if not path.is_file():
        print(f"Error: Video file not found: {path}")
        return

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
        fps = 30.0

    frame_delay = max(1, round(1000 / fps))

    print("=" * 60)
    print("REAL-TIME MOTION DETECTION (DIRECTION)")
    print("=" * 60)
    print(f"Video File      : {path.name}")
    print(f"Resolution      : {frame_width} x {frame_height}")
    print(f"Total Frames    : {total_frames}")
    print(f"Frame Rate      : {fps:.2f} FPS")
    print(f"Pixel Threshold : {pixel_threshold}")
    print("=" * 60)
    print()
    print("Controls:")
    print("  SPACE : Pause / Resume")
    print("  Q/ESC : Quit")
    print()

    # Read first frame
    success, prev_frame = video.read()
    if not success:
        print("Error: Could not read the first frame.")
        video.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    frame_number = 1
    paused = False

    # FPS calculation
    fps_update_interval = 0.5
    prev_tick = time.perf_counter()
    fps_counter = 0
    current_fps = 0.0

    # Motion level thresholds
    low_threshold = 5.0
    medium_threshold = 15.0

    # Optical flow parameters
    # Feature detection for tracking
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=15,
        blockSize=7,
    )

    # Lucas-Kanade optical flow parameters
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Tracked points
    prev_points = None
    # Color for arrows
    arrow_color = (0, 255, 0)  # Green

    try:
        while True:
            if not paused:
                success, frame = video.read()
                if not success:
                    print("\nEnd of video reached.")
                    break

                frame_number += 1
                fps_counter += 1

                # Calculate real-time FPS
                now = time.perf_counter()
                elapsed = now - prev_tick
                if elapsed >= fps_update_interval:
                    current_fps = fps_counter / elapsed
                    fps_counter = 0
                    prev_tick = now

                # --- Motion Detection ---
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                diff = cv2.absdiff(prev_gray, gray)

                # Motion score (0-255)
                motion_score = float(np.mean(diff))

                # Changed pixels mask
                changed_mask = diff > pixel_threshold
                changed_pct = float(np.count_nonzero(changed_mask) / diff.size * 100.0)

                # Classify motion level
                if motion_score < low_threshold and changed_pct < 10.0:
                    motion_level = "LOW"
                    level_color = (0, 255, 0)  # Green
                elif motion_score < medium_threshold:
                    motion_level = "MEDIUM"
                    level_color = (0, 255, 255)  # Yellow
                else:
                    motion_level = "HIGH"
                    level_color = (0, 0, 255)  # Red

                # --- Optical Flow for Direction Arrows ---
                # Detect good features to track in the previous frame
                if prev_points is None or len(prev_points) < 10:
                    prev_points = cv2.goodFeaturesToTrack(
                        prev_gray, mask=None, **feature_params
                    )

                if prev_points is not None and len(prev_points) > 0:
                    # Compute optical flow
                    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, prev_points, None, **lk_params
                    )

                    if curr_points is not None:
                        # Select good points
                        good_prev = prev_points[status == 1]
                        good_curr = curr_points[status == 1]

                        # Draw arrows showing motion direction
                        for i, (p0, p1) in enumerate(zip(good_prev, good_curr)):
                            x0, y0 = p0.ravel()
                            x1, y1 = p1.ravel()

                            # Calculate displacement
                            dx = x1 - x0
                            dy = y1 - y0
                            dist = np.sqrt(dx * dx + dy * dy)

                            # Only draw arrows for significant movement
                            if dist > 1.5:
                                # Scale arrow length for visibility
                                scale = 3.0
                                end_x = int(x0 + dx * scale)
                                end_y = int(y0 + dy * scale)

                                # Draw arrow
                                cv2.arrowedLine(
                                    frame,
                                    (int(x0), int(y0)),
                                    (end_x, end_y),
                                    arrow_color,
                                    2,
                                    tipLength=0.4,
                                )

                        # Update points for next iteration
                        prev_points = good_curr.reshape(-1, 1, 2)
                    else:
                        prev_points = None
                else:
                    prev_points = None

                # Update previous frame
                prev_gray = gray

            # --- Draw Overlay UI ---
            display = frame.copy()

            # Top-left: Video info
            cv2.putText(display, f"Frame: {frame_number}/{total_frames}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"FPS: {current_fps:.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Top-right: Motion level badge
            level_text = f"Motion: {motion_level}"
            (tw, th), _ = cv2.getTextSize(level_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(display, (frame_width - tw - 20, 10),
                          (frame_width - 10, 50), level_color, -1)
            cv2.putText(display, level_text,
                        (frame_width - tw - 15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 2)

            # Bottom-left: Motion score bar
            bar_x, bar_y = 10, frame_height - 50
            bar_w, bar_h = 300, 25
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          (50, 50, 50), -1)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          (200, 200, 200), 1)

            # Fill bar based on motion score (normalize 0-50 to 0-100%)
            fill_pct = min(1.0, motion_score / 50.0)
            fill_w = int(bar_w * fill_pct)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          level_color, -1)

            # Bar label
            cv2.putText(display, f"Motion Score: {motion_score:.1f}",
                        (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

            # Bottom-right: Changed pixels %
            cv2.putText(display, f"Changed: {changed_pct:.1f}%",
                        (frame_width - 200, frame_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Pause indicator
            if paused:
                cv2.putText(display, "PAUSED",
                            (frame_width // 2 - 60, frame_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            cv2.imshow("Real-Time Motion Detection", display)

            # Handle key presses
            delay = 30 if paused else frame_delay
            key = cv2.waitKey(delay) & 0xFF

            if key == ord("q") or key == 27:
                print("\nQuit by user.")
                break
            if key == ord(" "):
                paused = not paused
                print("Paused" if paused else "Resumed")

    finally:
        video.release()
        cv2.destroyAllWindows()
        print("\nVideo closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time motion detection with direction arrows."
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=r"C:\Users\tevee\Downloads\exp31.mp4",
        help="Path to the video file",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=25,
        help="Pixel intensity threshold for motion (default: 25)",
    )

    args = parser.parse_args()

    run_live_motion_detection(args.video_path, args.threshold)