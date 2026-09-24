"""
Traffic Video Object Detection using OpenCV Background Subtraction (MOG2).

This script detects moving vehicles/objects in a traffic video using
OpenCV's background subtraction (MOG2) algorithm. It samples every Nth
frame, filters contours by minimum area, draws bounding boxes, and
displays the results in a live window.

Usage:
    python traffic_detection.py --video path/to/video.mp4
    python traffic_detection.py --video traffic.mp4 --interval 20 --min-area 500
"""

import argparse
import sys

import cv2
import numpy as np


# =========================================================
# Configuration
# =========================================================
DEFAULT_INTERVAL = 20       # Process every Nth frame
DEFAULT_MIN_AREA = 500      # Minimum contour area to count as an object
MOG2_HISTORY = 500          # Background subtractor history length
MOG2_VAR_THRESHOLD = 50     # Background subtractor variance threshold
MOG2_DETECT_SHADOWS = True  # Detect and mark shadows
THRESHOLD_VALUE = 200       # Binary threshold value for shadow removal
MORPH_KERNEL_SIZE = 5       # Morphological kernel size
MORPH_DILATE_ITERATIONS = 2 # Dilation iterations


# =========================================================
# 1. Load Traffic Video
# =========================================================
def load_video(video_path):
    """
    Open a video file and return a VideoCapture object.

    Args:
        video_path (str): Path to the video file.

    Returns:
        cv2.VideoCapture: Opened video capture object.

    Raises:
        FileNotFoundError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video file: {video_path}\n"
            "Please check that the file exists and is a valid video format."
        )
    return cap


# =========================================================
# 2. Display Video Properties
# =========================================================
def display_video_properties(cap, video_path):
    """
    Print video metadata (resolution, FPS, frame count, duration).

    Args:
        cap (cv2.VideoCapture): Opened video capture object.
        video_path (str): Path to the video file (for display only).
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("========================================")
    print("        VIDEO PROPERTIES")
    print("========================================")
    print("Video        :", video_path)
    print("Resolution   :", width, "x", height)
    print("FPS          :", fps)
    print("Total Frames :", total_frames)
    if fps > 0:
        duration = total_frames / fps
        print("Duration     :", round(duration, 2), "seconds")


# =========================================================
# 3. Background Subtractor
# =========================================================
def create_background_subtractor():
    """
    Create and return a MOG2 background subtractor.

    Returns:
        cv2.BackgroundSubtractorMOG2: Configured MOG2 subtractor.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS
    )


# =========================================================
# 4. Detect Moving Objects
# =========================================================
def detect_objects(frame, background_subtractor, min_area):
    """
    Detect moving objects in a single frame using background subtraction.

    Args:
        frame (np.ndarray): Input BGR frame.
        background_subtractor (cv2.BackgroundSubtractorMOG2): MOG2 subtractor.
        min_area (int): Minimum contour area to count as an object.

    Returns:
        tuple: (list of bounding boxes, list of object labels)
            - boxes: List of (x, y, w, h) tuples for each detected object.
            - labels: List of label strings like "Object 1", "Object 2", ...
    """
    # Apply background subtraction
    foreground_mask = background_subtractor.apply(frame)

    # Remove shadows (MOG2 marks shadows with value 127)
    _, threshold = cv2.threshold(
        foreground_mask,
        THRESHOLD_VALUE,
        255,
        cv2.THRESH_BINARY
    )

    # Remove small noise with morphological opening
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill gaps with dilation
    threshold = cv2.dilate(
        threshold,
        kernel,
        iterations=MORPH_DILATE_ITERATIONS
    )

    # Find contours
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    labels = []
    object_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        # Ignore very small objects
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        object_count += 1
        boxes.append((x, y, w, h))
        labels.append(f"Object {object_count}")

    return boxes, labels


# =========================================================
# 5. Draw Annotations
# =========================================================
def draw_annotations(frame, boxes, labels, frame_number):
    """
    Draw bounding boxes, object labels, and info text on the frame.

    Args:
        frame (np.ndarray): Input BGR frame (modified in place).
        boxes (list): List of (x, y, w, h) bounding boxes.
        labels (list): List of label strings.
        frame_number (int): Current frame number for display.
    """
    # Draw bounding boxes and object numbers
    for (x, y, w, h), label in zip(boxes, labels):
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        # Object number
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display object count
    cv2.putText(
        frame,
        f"Objects Detected: {len(boxes)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Display frame number
    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


# =========================================================
# 6. Process Frame
# =========================================================
def process_frame(frame, background_subtractor, min_area, frame_number):
    """
    Process a single frame: detect objects, draw annotations, display.

    Args:
        frame (np.ndarray): Input BGR frame.
        background_subtractor (cv2.BackgroundSubtractorMOG2): MOG2 subtractor.
        min_area (int): Minimum contour area to count as an object.
        frame_number (int): Current frame number.

    Returns:
        int: Number of objects detected in this frame.
    """
    print("\n----------------------------------------")
    print("Frame Number:", frame_number)

    # Detect moving objects
    boxes, labels = detect_objects(frame, background_subtractor, min_area)

    # Draw annotations
    draw_annotations(frame, boxes, labels, frame_number)

    # Print number of objects
    print("Number of Objects Detected:", len(boxes))

    # Display frame
    cv2.imshow(
        "Traffic Video - Object Detection",
        frame
    )

    return len(boxes)


# =========================================================
# 7. Main Processing Loop
# =========================================================
def run_detection(video_path, interval, min_area):
    """
    Run the main video processing loop.

    Args:
        video_path (str): Path to the video file.
        interval (int): Process every Nth frame.
        min_area (int): Minimum contour area to count as an object.

    Returns:
        int: Exit code (0 on success, 1 on error).
    """
    try:
        # Load video
        cap = load_video(video_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    try:
        # Display video properties
        display_video_properties(cap, video_path)

        # Create background subtractor
        background_subtractor = create_background_subtractor()

        # Process video
        frame_number = 0
        while True:
            ret, frame = cap.read()

            # Handle corrupted or missing frames
            if not ret:
                print("\nWarning: Reached end of video or encountered a corrupted frame.")
                break

            # Process every Nth frame
            if frame_number % interval == 0:
                process_frame(
                    frame,
                    background_subtractor,
                    min_area,
                    frame_number
                )

                # Press q to quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nQuit requested by user.")
                    break

            frame_number += 1

    except cv2.error as e:
        print(f"\nOpenCV error: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    print("\n========================================")
    print("Video processing completed successfully.")
    print("========================================")
    return 0


# =========================================================
# 8. CLI Argument Parsing
# =========================================================
def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Detect moving objects in a traffic video using "
                    "OpenCV background subtraction (MOG2)."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file (e.g., traffic.mp4)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Process every Nth frame (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help=f"Minimum contour area to count as an object "
             f"(default: {DEFAULT_MIN_AREA})"
    )
    return parser.parse_args()


# =========================================================
# 9. Entry Point
# =========================================================
def main():
    """
    Main entry point for the script.
    """
    args = parse_args()
    exit_code = run_detection(
        video_path=args.video,
        interval=args.interval,
        min_area=args.min_area
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()