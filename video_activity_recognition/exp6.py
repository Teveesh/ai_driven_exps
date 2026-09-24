"""
Experiment 06 - CNN-Based Video Feature Extraction
====================================================
A video analytics application that, for a given video file:

  1. Loads the video with OpenCV and prints its properties
     (resolution, FPS, total frame count).
  2. Samples frames at a configurable interval (default: every 20th frame).
  3. Runs each sampled frame through a pretrained CNN (torchvision
     ResNet18 / MobileNetV2 with ImageNet weights), removing the final
     classification layer, to obtain a compact feature embedding.
  4. Detects moving objects per frame with MOG2 background subtraction
     (contours filtered by area -> bounding boxes / object counts).
  5. Saves the CNN feature vectors to disk (.npy) indexed by frame number
     and timestamp.
  6. Annotates and displays each processed frame with bounding boxes,
     object count, and frame number, plus a console summary per frame.
  7. Prints a final summary (processed frames, total objects, file paths).

Usage:
    python exp6.py --video traffic.mp4 --interval 20 --model resnet18
"""

import argparse
import os
import sys

import cv2
import numpy as np


# ==========================================================================
# Configuration
# ==========================================================================
# Path to the input video. By default it is looked up next to this script so
# the app works regardless of the current working directory.
VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traffic.mp4")

FRAME_INTERVAL = 20      # process every Nth frame (configurable)
CNN_MODEL_NAME = "resnet18"   # "resnet18" or "mobilenet_v2"
USE_IMAGENET_WEIGHTS = True   # False -> randomly initialized weights

# ImageNet preprocessing constants for torchvision models.
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Output directory / files.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FEATURES_FILE = os.path.join(OUTPUT_DIR, "cnn_features.npy")        # (N, D)
FRAME_INDEX_FILE = os.path.join(OUTPUT_DIR, "cnn_frame_indices.npy")  # (N,)
TIMESTAMP_FILE = os.path.join(OUTPUT_DIR, "cnn_timestamps.npy")      # (N,)

# Display the annotated frames in a window (0 disables for headless runs).
SHOW_WINDOWS = os.environ.get("EXP6_SHOW_WINDOWS", "1") != "0"

# Object-detection parameters (preserved from the existing MOG2-based
# motion-detection / bounding-box drawing logic).
MIN_AREA = 500               # minimum contour area to count as an object
MOG2_HISTORY = 500           # background subtractor history length
MOG2_VAR_THRESHOLD = 50      # background subtractor variance threshold
MOG2_DETECT_SHADOWS = True   # MOG2 marks shadows with value 127
THRESHOLD_VALUE = 200        # binary threshold for shadow removal
MORPH_KERNEL_SIZE = 5        # morphological kernel size
MORPH_DILATE_ITERATIONS = 2  # dilation iterations

# Motion-level thresholds (preserved from the motion-detection logic).
LOW_MOTION_THRESHOLD = 5.0
MEDIUM_MOTION_THRESHOLD = 15.0
CHANGED_PIXEL_THRESHOLD = 10.0
PIXEL_DIFF_THRESHOLD = 25


# ==========================================================================
# 1. Load Video
# ==========================================================================
def load_video(video_path):
    """
    Open a video file and return a VideoCapture object.

    Args:
        video_path (str): Path to the video file.

    Returns:
        cv2.VideoCapture: Opened video capture object.

    Raises:
        FileNotFoundError: If the file does not exist or cannot be opened.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}\n"
            "Place the video in the project folder or pass --video."
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(
            f"Could not open video file: {video_path}\n"
            "Please check that the file is a valid video format."
        )
    return cap


# ==========================================================================
# 2. Video Properties
# ==========================================================================
def get_video_properties(cap):
    """
    Extract video metadata from an opened capture object and print it.

    Args:
        cap (cv2.VideoCapture): Opened video capture object.

    Returns:
        dict: {"width", "height", "fps", "total_frames", "duration"}
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0
    duration = total_frames / fps if total_frames > 0 else 0.0

    print("==========================================")
    print("        VIDEO PROPERTIES")
    print("==========================================")
    print(f"Resolution   : {width} x {height}")
    print(f"FPS          : {fps:.2f}")
    print(f"Total Frames : {total_frames}")
    print(f"Duration     : {duration:.2f} seconds")
    print("==========================================")

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
    }
# ==========================================================================
# 3. Pretrained CNN (torchvision) - Feature Extractor
# ==========================================================================
def build_cnn(model_name, use_imagenet):
    """
    Load a torchvision CNN and strip the final classification layer so the
    forward pass returns the penultimate-layer feature embedding.

    Args:
        model_name (str): "resnet18" or "mobilenet_v2".
        use_imagenet (bool): Load ImageNet pretrained weights if True.

    Returns:
        tuple: (model in eval() mode, feature dimension)
    """
    import torch
    import torchvision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "resnet18":
        weights_enum = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    elif model_name == "mobilenet_v2":
        weights_enum = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
    else:
        raise ValueError(
            f"Unknown CNN model '{model_name}'. Use 'resnet18' or 'mobilenet_v2'."
        )

    # Try to load pretrained weights. If the download fails (no network),
    # fall back to random initialization with a clear warning.
    try:
        if use_imagenet:
            print(f"[CNN] Loading ImageNet pretrained weights for {model_name} ...")
            if model_name == "resnet18":
                model = torchvision.models.resnet18(weights=weights_enum)
            else:
                model = torchvision.models.mobilenet_v2(weights=weights_enum)
        else:
            print(f"[CNN] Using randomly initialized {model_name} (no pretrained weights).")
            if model_name == "resnet18":
                model = torchvision.models.resnet18(weights=None)
            else:
                model = torchvision.models.mobilenet_v2(weights=None)
    except Exception as exc:  # e.g. URLError/ConnectionError during download
        print(f"[CNN] WARNING: could not load pretrained weights ({exc}).")
        print("[CNN] Falling back to randomly initialized weights.")
        if model_name == "resnet18":
            model = torchvision.models.resnet18(weights=None)
        else:
            model = torchvision.models.mobilenet_v2(weights=None)

    # Remove the classification head -> intermediate/penultimate features.
    if model_name == "resnet18":
        model.fc = torch.nn.Identity()          # 512-D features
        feature_dim = 512
    else:
        model.classifier[1] = torch.nn.Identity()  # 1280-D features
        feature_dim = 1280

    model.eval().to(device)
    print(f"[CNN] {model_name} ready | feature dimension = {feature_dim} "
          f"| device = {device}")
    return model, feature_dim, device


# ==========================================================================
# 4. Frame Preprocessing + CNN Feature Extraction
# ==========================================================================
def preprocess_frame(frame):
    """
    Convert an OpenCV BGR frame into an ImageNet-normalized 4D tensor.

    Args:
        frame (np.ndarray): BGR frame.

    Returns:
        torch.Tensor: shape (1, 3, IMG_SIZE, IMG_SIZE) on the current device,
        or None if torch is unavailable.
    """
    import torch

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    normalized = (resized - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
        IMAGENET_STD, dtype=np.float32
    )
    # HWC -> CHW, add batch dimension.
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor


def extract_frame_features(model, frame, device=None):
    """
    Run one frame through the CNN and return the feature embedding.

    Args:
        model (torch.nn.Module): CNN with the classification head removed.
        frame (np.ndarray): BGR frame to process.
        device (torch.device, optional): Inference device.

    Returns:
        np.ndarray: 1-D feature vector of length feature_dim (float32).
    """
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor = preprocess_frame(frame).to(device)
    with torch.no_grad():          # no gradients needed for inference
        embedding = model(tensor)  # (1, feature_dim)

    return embedding.squeeze(0).cpu().numpy().astype(np.float32)
# ==========================================================================
# 5. Background Subtractor (preserved MOG2 configuration)
# ==========================================================================
def create_background_subtractor():
    """
    Create and return a MOG2 background subtractor.

    Returns:
        cv2.BackgroundSubtractorMOG2: Configured MOG2 subtractor.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS,
    )


# ==========================================================================
# 6. Detect Moving Objects
# ==========================================================================
def detect_objects(frame, background_subtractor, min_area=MIN_AREA):
    """
    Detect moving objects in a single frame using MOG2 background subtraction
    (this preserves the existing motion-detection / bounding-box logic).

    Args:
        frame (np.ndarray): Input BGR frame.
        background_subtractor (cv2.BackgroundSubtractorMOG2): MOG2 subtractor.
        min_area (int): Minimum contour area to count as an object.

    Returns:
        tuple: (boxes, labels)
            - boxes: List of (x, y, w, h) tuples for each detected object.
            - labels: List of label strings like "Object 1", "Object 2", ...
    """
    # Apply background subtraction.
    foreground_mask = background_subtractor.apply(frame)

    # Remove shadows (MOG2 marks shadows with value 127).
    _, threshold = cv2.threshold(
        foreground_mask, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY
    )

    # Remove small noise with morphological opening.
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    # Fill gaps with dilation.
    threshold = cv2.dilate(
        threshold, kernel, iterations=MORPH_DILATE_ITERATIONS
    )

    # Find contours.
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    labels = []
    object_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:     # ignore very small objects
            continue
        x, y, w, h = cv2.boundingRect(contour)
        object_count += 1
        boxes.append((x, y, w, h))
        labels.append(f"Object {object_count}")

    return boxes, labels


# ==========================================================================
# 7. Motion Score (preserved frame-differencing logic)
# ==========================================================================
def compute_motion_score(gray, prev_gray):
    """
    Compute a motion score between the current and previous (processed) frame
    using the existing motion-detection logic (Gaussian blur + absdiff).

    Args:
        gray (np.ndarray): Current grayscale frame.
        prev_gray (np.ndarray): Previous grayscale frame (or None).

    Returns:
        tuple: (mad_score, changed_pct, motion_level)
    """
    if prev_gray is None:
        return 0.0, 0.0, "Low"

    diff = cv2.absdiff(prev_gray, gray)
    mad = float(np.mean(diff))                                    # 0-255 scale
    changed_pct = float(np.count_nonzero(diff > PIXEL_DIFF_THRESHOLD)
                        / diff.size * 100.0)

    if mad < LOW_MOTION_THRESHOLD and changed_pct < CHANGED_PIXEL_THRESHOLD:
        level = "Low"
    elif mad < MEDIUM_MOTION_THRESHOLD:
        level = "Medium"
    else:
        level = "High"

    return mad, changed_pct, level


# ==========================================================================
# 8. Draw Annotations
# ==========================================================================
def draw_annotations(frame, boxes, labels, frame_number, fps, feature_dim):
    """
    Draw bounding boxes, labels, object count, frame number, timestamp and the
    CNN feature dimension on the frame (in place).

    Args:
        frame (np.ndarray): Input/annotated BGR frame.
        boxes (list): List of (x, y, w, h) bounding boxes.
        labels (list): List of label strings.
        frame_number (int): Current frame number.
        fps (float): Video frame rate (for the timestamp overlay).
        feature_dim (int): CNN feature vector length (informational overlay).
    """
    for (x, y, w, h), label in zip(boxes, labels):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    # Object count (red).
    cv2.putText(
        frame, f"Objects Detected: {len(boxes)}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
    )
    # Frame number (blue).
    cv2.putText(
        frame, f"Frame: {frame_number}", (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2,
    )
    # Timestamp (white).
    cv2.putText(
        frame, f"Time: {frame_number / fps:.2f}s", (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    # CNN feature info (magenta).
    cv2.putText(
        frame, f"CNN Features: {feature_dim}-D", (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2,
    )
# ==========================================================================
# 9. Save CNN Features
# ==========================================================================
def save_features(features, frame_indices, timestamps):
    """Save CNN feature vectors + frame indices/timestamps as .npy files.

    Args:
        features (list[np.ndarray]): One embedding per processed frame.
        frame_indices (list[int]): Frame numbers of processed frames.
        timestamps (list[float]): Timestamps of processed frames.

    Returns:
        str: Path to the saved features file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    feature_matrix = np.vstack(features).astype(np.float32)  # (N, feature_dim)
    index_array = np.asarray(frame_indices, dtype=np.int64)  # (N,)
    time_array = np.asarray(timestamps, dtype=np.float32)    # (N,)

    np.save(FEATURES_FILE, feature_matrix)
    np.save(FRAME_INDEX_FILE, index_array)
    np.save(TIMESTAMP_FILE, time_array)

    print(f"\n[SAVE] Feature matrix saved : {FEATURES_FILE}")
    print(f"       - shape: {feature_matrix.shape} (frames x feature_dim)")
    print(f"[SAVE] Frame indices saved  : {FRAME_INDEX_FILE}")
    print(f"[SAVE] Timestamps saved     : {TIMESTAMP_FILE}")
    return FEATURES_FILE


# ==========================================================================
# 10. Main Processing Loop
# ==========================================================================
def run_pipeline(video_path=VIDEO_PATH, interval=FRAME_INTERVAL,
                 model_name=CNN_MODEL_NAME, min_area=MIN_AREA):
    """Run the full extraction pipeline over a video.

    Args:
        video_path (str): Path to the input video.
        interval (int): Process every Nth frame.
        model_name (str): CNN architecture to use.
        min_area (int): Minimum contour area for object detection.

    Returns:
        int: Exit code (0 on success, 1 on error).
    """
    cap = None
    try:
        cap = load_video(video_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        info = get_video_properties(cap)          # resolution / fps / frames
        fps = info["fps"]
        total_frames = info["total_frames"]

        # Build the pretrained CNN feature extractor.
        model, feature_dim, device = build_cnn(model_name, USE_IMAGENET_WEIGHTS)

        # Background subtractor for object detection.
        background_subtractor = create_background_subtractor()

        features = []        # CNN embeddings per processed frame
        frame_indices = []   # frame number per processed frame
        timestamps = []      # timestamp per processed frame
        total_objects = 0    # objects detected across all processed frames
        processed = 0        # number of frames actually processed

        prev_gray = None     # grayscale of the previous *processed* frame
        frame_number = 0

        print(f"\nProcessing every {interval}th frame... (press 'q' to quit)")

        while True:
            ret, frame = cap.read()
            if not ret:      # corrupted frame or end of video
                print("\nWarning: reached the end of the video or "
                      "encountered a corrupted frame.")
                break

            if frame_number % interval != 0:
                frame_number += 1
                continue

            timestamp = frame_number / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # 1) Motion score (preserved frame-differencing logic).
            mad, changed_pct, motion_level = compute_motion_score(gray, prev_gray)

            # 2) Object detection (preserved MOG2 + bounding-box logic).
            boxes, labels = detect_objects(frame, background_subtractor,
                                           min_area)
            total_objects += len(boxes)

            # 3) CNN feature extraction (pretrained torchvision model).
            embedding = extract_frame_features(model, frame, device)
            features.append(embedding)
            frame_indices.append(frame_number)
            timestamps.append(timestamp)

            # 4) Annotate the frame: boxes, count, frame number, timestamp.
            annotated = frame.copy()
            draw_annotations(annotated, boxes, labels, frame_number, fps,
                             feature_dim)

            # 5) Console summary for this frame.
            print("\n----------------------------------------")
            print(f"Frame Number        : {frame_number}")
            print(f"Timestamp           : {timestamp:.2f} s")
            print(f"Objects Detected    : {len(boxes)}")
            print(f"Bounding Boxes      : {boxes}")
            print(f"Motion Score (MAD)  : {mad:.2f}  [{motion_level}]")
            print(f"Changed Pixels      : {changed_pct:.2f}%")
            print(f"CNN Feature Vector  : {embedding.shape[0]}-D "
                  f"(first 8 values: {np.round(embedding[:8], 4)})")

            # 6) Display the annotated frame.
            if SHOW_WINDOWS:
                cv2.imshow("Exp 06 - CNN Video Feature Extraction", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nQuit requested by user.")
                    break

            prev_gray = gray
            processed += 1
            frame_number += 1
# ------------------------------------------------------------------
        # Final summary
        # ------------------------------------------------------------------
        print("\n==========================================")
        print("        FINAL SUMMARY")
        print("==========================================")
        print(f"Frames Processed       : {processed} "
              f"(every {interval}th of {total_frames})")
        print(f"Total Objects Detected : {total_objects}")
        print(f"Features Saved To      : {FEATURES_FILE}")
        print("==========================================")

        if processed > 0:
            save_features(features, frame_indices, timestamps)
        else:
            print("No frames were processed - nothing saved.")

        return 0

    except cv2.error as exc:
        print(f"\nOpenCV error: {exc}")
        return 1
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        return 1
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


# ==========================================================================
# 11. CLI Argument Parsing
# ==========================================================================
def parse_args():
    """Parse command-line arguments (override the config defaults)."""
    parser = argparse.ArgumentParser(
        description="CNN-based video feature extraction with object detection."
    )
    parser.add_argument("--video", type=str, default=VIDEO_PATH,
                        help=f"Path to the input video (default: {VIDEO_PATH})")
    parser.add_argument("--interval", type=int, default=FRAME_INTERVAL,
                        help=f"Process every Nth frame (default: {FRAME_INTERVAL})")
    parser.add_argument("--model", type=str, default=CNN_MODEL_NAME,
                        choices=["resnet18", "mobilenet_v2"],
                        help=f"CNN architecture (default: {CNN_MODEL_NAME})")
    parser.add_argument("--min-area", type=int, default=MIN_AREA,
                        help=f"Min contour area for detection (default: {MIN_AREA})")
    return parser.parse_args()


# ==========================================================================
# 12. Entry Point
# ==========================================================================
def main():
    """Main entry point for the script."""
    args = parse_args()
    exit_code = run_pipeline(
        video_path=args.video,
        interval=args.interval,
        model_name=args.model,
        min_area=args.min_area,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()