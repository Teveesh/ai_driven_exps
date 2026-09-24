# AI-Driven Video Processing Experiments

A collection of OpenCV / PyTorch / TensorFlow experiments for **video processing,
motion analysis, object detection and activity recognition**.

Each script is a self-contained experiment with a documented header (purpose,
structure and usage) so it can be read and run on its own.

## Repository layout

```
aidriven/
|
|-- video_properties.py           # Inspect video metadata + player with frame/FPS overlay
|-- extract_frames.py             # Extract frames from a video to extracted_frames/
|-- motion_detection.py           # Off-line motion analysis (CSV report + heatmap)
|-- motion_detection_live.py      # Real-time motion detection with direction arrows
|-- library_check.py              # Print installed library versions and GPU availability
|
|-- motion_reports/               # Motion reports (.csv) and heatmaps (.png)
|
|-- traffic_object_detection/     # Experiment: MOG2 traffic object detection
|   |-- traffic_object_detection.py  # MOG2 + threshold + morphology + contours
|   |-- traffic.mp4               # Sample video (not tracked, see .gitignore)
|   |-- README.md                 # Aim, algorithm and explanations (lab report)
|   |-- requirements.txt
|
|-- video_activity_recognition/   # Experiments 06 and 07
|   |-- exp6.py                   # CNN feature extraction + MOG2 object detection
|   |-- 7exp.py                   # Activity recognition with NumPy RNN and LSTM
|   |-- traffic.mp4               # Sample input video
|   |-- requirements.txt
|   `-- outputs/                  # Experiment results (.png / .npy)
|
`-- traffic_detection_project/    # Standalone traffic detection project
    |-- traffic_detection.py      # MOG2 background-subtraction vehicle detection
    `-- requirements.txt
```

## Scripts

### `video_properties.py`
Opens a video and prints its properties (name, resolution, frame count, FPS,
duration), then plays it in a window with the current frame number and a live
FPS counter drawn on top.

Controls: `Space` pause / resume, `Q` or `Esc` quit.

### `extract_frames.py`
Extracts frames from a video to image files.

```bash
python extract_frames.py "path/to/video.mp4"
python extract_frames.py "path/to/video.mp4" --output frames --interval 5 --skip
python extract_frames.py "path/to/video.mp4" --start 100 --end 500
```

| Option | Description |
| --- | --- |
| `--output` / `-o` | Output directory (default `extracted_frames`) |
| `--interval` / `-i` | Keep every Nth frame (default 1) |
| `--start` / `-s` | First frame index (default 0) |
| `--end` / `-e` | Last frame index (default: end of video) |
| `--skip` | Use interval-based extraction instead of saving every frame |

### `motion_detection.py`
Off-line motion analysis using frame differencing. For every pair of
consecutive frames it computes:

- a motion score (mean absolute difference of pixel intensities, 0-255),
- the percentage of changed pixels,
- a motion level (`Low` / `Medium` / `High`),

and finally prints overall statistics plus a motion timeline. With
`--report` it also writes a CSV report and a motion heatmap image.

```bash
python motion_detection.py "path/to/video.mp4"
python motion_detection.py "path/to/video.mp4" --threshold 25 --report
```

Motion-level thresholds: MAD `< 5.0` and changed pixels `< 10 %` is `Low`,
MAD `< 15.0` is `Medium`, otherwise `High`.

### `motion_detection_live.py`
Plays a video while performing real-time motion detection with optical flow.
Green arrows show the motion direction, together with a live motion score bar,
the current motion level, a frame counter and the FPS.

Controls: `Space` pause / resume, `Q` or `Esc` quit.

### `library_check.py`
Prints the installed versions of OpenCV, TensorFlow, PyTorch and Transformers,
plus the available TensorFlow GPU devices, PyTorch CUDA availability and the
GPU name.

### `video_activity_recognition/exp6.py` — Experiment 06
CNN-based video feature extraction:

1. loads the video with OpenCV and prints its properties,
2. samples frames at a configurable interval (default: every 20th frame),
3. runs each sampled frame through a pretrained torchvision CNN
   (`resnet18` or `mobilenet_v2`, ImageNet weights) with the classifier
   removed, producing a compact feature embedding,
4. detects moving objects per frame with MOG2 background subtraction
   (contours filtered by area -> bounding boxes / object counts),
5. saves the feature vectors as `.npy` files indexed by frame number and
   timestamp,
6. annotates and displays each processed frame, and prints a final summary.

```bash
python exp6.py --video traffic.mp4 --interval 20 --model resnet18
```

### `video_activity_recognition/7exp.py` — Experiment 07
Activity recognition (running / fast movement vs. walking / normal movement)
from temporal visual features, using a SimpleRNN and a SimpleLSTM implemented
manually with NumPy (no deep-learning framework):

```
traffic.mp4
  -> 20 equally spaced frames (BGR -> RGB)
  -> visual features (mean, standard deviation, edge density, motion)
  -> min-max normalization
  -> 64-D feature sequence
  -> SimpleRNN  (20, 64) -> (20, 32)
  -> SimpleLSTM (20, 64) -> (20, 32)
  -> activity classification + visualization + saved .npy features (outputs/)
```

### `traffic_detection_project/traffic_detection.py`
Detects moving vehicles / objects in a traffic video with OpenCV MOG2
background subtraction: samples every Nth frame, filters contours by minimum
area, draws bounding boxes and shows the result in a live window.

```bash
python traffic_detection.py --video path/to/video.mp4
python traffic_detection.py --video traffic.mp4 --interval 20 --min-area 500
```

### `traffic_object_detection/traffic_object_detection.py`
College-laboratory implementation of traffic object detection with fixed
parameters: resize to 800 x 600, MOG2 (`history=500`, `varThreshold=50`,
`detectShadows=True`), binary threshold at 200, 5 x 5 opening followed by 2
dilations, contours with area > 500, green boxes with object numbers. Press
`Q` to stop; at the end it prints the total frames, average objects, average
time per frame and processing FPS.

```bash
cd traffic_object_detection

# traffic.mp4 is a video file and is therefore not tracked by git;
# copy the sample clip that is already in this repository:
copy ..\video_activity_recognition\traffic.mp4 .

python traffic_object_detection.py
```

`traffic_object_detection/README.md` contains the aim, the algorithm and the
explanations of MOG2, thresholding, morphology and contours.

## Setup

The scripts share one Python 3.12 virtual environment at the repository root.

```powershell
python -m venv ai_video_env
ai_video_env\Scripts\activate
pip install opencv-python numpy matplotlib torch torchvision
python library_check.py
```

`tensorflow` and `transformers` are only needed by `library_check.py`;
`matplotlib` is used by `7exp.py`.

## Notes

The virtual environments (`ai_video_env/`, `Lib/`, `Scripts/`,
`traffic_detection_project/venv/`), the generated frames
(`extracted_frames/`) and the sample video
`traffic_object_detection/traffic.mp4` (7 MB - copy it from
`video_activity_recognition/` before running that experiment) are
intentionally **not** tracked - see `.gitignore`.
Install the dependencies with the commands above (or the per-project
`requirements.txt` files) and regenerate the frames with `extract_frames.py`.

## Requirements

- Python 3.12
- OpenCV (`opencv-python`), NumPy, Matplotlib
- PyTorch + torchvision (CNN feature extraction)
- TensorFlow + Transformers (environment check only)
