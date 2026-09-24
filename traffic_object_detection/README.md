# Experiment: Traffic Object Detection using Background Subtraction

**File:** `traffic_object_detection.py` | **Input video:** `traffic.mp4` | **Language:** Python + OpenCV (`cv2`)

## Aim

To detect and count the moving objects (vehicles) in a traffic video using
OpenCV **background subtraction (MOG2)** and to mark every detected object with
a **green bounding box** and its **object number**, displaying the object count
and the frame number on the video.

## Requirements

**Software**

- Python 3.x
- OpenCV: `pip install opencv-python` (see `requirements.txt`)
- Input video `traffic.mp4` (in the same folder / working directory)

**Hardware**

- Any computer with a display (no GPU and no camera are needed - the input is a video file)

## Files

| File | Description |
| --- | --- |
| `traffic_object_detection.py` | The experiment (one self-contained script) |
| `traffic.mp4` | Input traffic video - **not tracked** (video file, see `.gitignore`); copy it from `../video_activity_recognition/traffic.mp4` |
| `requirements.txt` | Python dependency (`opencv-python`) |

## How to run

```bash
cd traffic_object_detection

# traffic.mp4 is not stored in the repository (it is a video file), so copy
# the sample clip that is already in the repository into this folder:
copy ..\video_activity_recognition\traffic.mp4 .

python traffic_object_detection.py
```

The script opens the video with `cv2.VideoCapture("traffic.mp4")`, so
`traffic.mp4` must be in the current working directory. Press **Q** in the
window to stop early; otherwise the script ends by itself after the last frame
and prints the results.

## Algorithm / Steps

1. Open the video with `cv2.VideoCapture("traffic.mp4")`.
2. If the video could not be opened, print an error message and exit.
3. Create the MOG2 background subtractor with `history=500`,
   `varThreshold=50` and `detectShadows=True`.
4. Create a 5 x 5 rectangular kernel with `cv2.getStructuringElement`
   (`cv2.MORPH_RECT`).
5. Read the video **frame by frame** and for every frame:
   1. Resize the frame to **800 x 600** pixels (`cv2.resize`).
   2. Apply the background subtractor (`back_sub.apply`) - this gives a
      grey-scale foreground mask.
   3. **Binary threshold at 200** (`cv2.threshold`, `THRESH_BINARY`) to remove
      the shadows and the weak noise.
   4. **Morphological opening** with the 5 x 5 kernel (`cv2.morphologyEx`,
      `MORPH_OPEN`) to remove the small noise blobs.
   5. **Dilation** with the same kernel, **2 iterations** (`cv2.dilate`), to
      make the objects solid.
   6. Find the **contours** of the white objects (`cv2.findContours`,
      `RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`).
   7. Ignore every contour with `cv2.contourArea(contour) <= 500`
      (these are noise, not vehicles).
   8. For every remaining object: compute its bounding box
      (`cv2.boundingRect`), draw it in green (`cv2.rectangle`) and write the
      object number on it (`cv2.putText`).
   9. Display the **number of objects** and the **current frame number** on
      the frame.
   10. Show the frame in the window titled `Traffic Object Detection`
       (`cv2.imshow`).
   11. Wait 30 ms for a key (`cv2.waitKey`); if the key is `q` or `Q`, stop
       the loop.
6. Release the video (`cap.release()`) and close the OpenCV windows
   (`cv2.destroyAllWindows()`).
7. Calculate and print the final statistics: **total frames**,
   **average objects**, **average time/frame** and **processing FPS**.

## Explanation of MOG2 (Mixture of Gaussians version 2)

MOG2 is an **adaptive background subtraction** algorithm. For every pixel of the
video it keeps a small number of Gaussian distributions that model the colours
that pixel has shown over time ("history"). When a new frame arrives, the
algorithm checks whether the current colour of the pixel fits those models:

- if it fits, the pixel is considered **background** (it becomes black in the mask),
- if it does not fit, the pixel is considered **foreground**, i.e. a moving object (it becomes white).

Parameters used here:

| Parameter | Value | Meaning |
| --- | --- | --- |
| `history` | 500 | Number of previous frames used to learn the background. A longer history = more stable background. |
| `varThreshold` | 50 | Squared-distance threshold. A pixel is foreground when its distance from the learned background is greater than this value. A small value detects more (also noise), a large value detects less. |
| `detectShadows` | `True` | Shadows are also detected, but they are **not** marked white - they are marked grey (about value 127). |

MOG2 is used instead of simple frame differencing because the background of a
traffic video is not constant (light changes, camera noise, trees moving), and
MOG2 learns and updates the background automatically.

## Explanation of Thresholding

`cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)` examines every pixel of the
MOG2 mask:

- pixel value **> 200** -> becomes **255** (pure white = object)
- pixel value **<= 200** -> becomes **0** (black = background)

A threshold of 200 is chosen for two reasons:

1. **Shadow removal** - with `detectShadows=True`, MOG2 writes the shadows in
   grey (about 127). Since 127 < 200, all the shadow pixels are turned black, so
   a vehicle's shadow is not detected as a separate object.
2. **Noise removal** - weak/uncertain foreground pixels (values between 1 and
   200) are also discarded, so only strong, confident foreground stays white.

Binary thresholding gives a clean black/white image that the morphological
operations and `findContours` can work on.

## Explanation of Morphological Opening and Dilation

Both operations use the same **5 x 5 rectangular kernel**
(`cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))`), which is a small window
that slides over the binary image.

**Morphological opening** (`cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)`) is
an **erosion followed by a dilation**:

- *Erosion* shrinks the white regions, so small white dots (noise, a few pixels
  wide) disappear completely.
- *Dilation* then grows the remaining regions back to (almost) their original
  size.
- Net effect: **small noise blobs are removed while the real objects keep their
  size.**

**Dilation** (`cv2.dilate(mask, kernel, iterations=2)`) grows the white regions.
Each iteration extends the white area by the size of the kernel. It is applied
**2 times** here so that:

- parts of the same object that were separated (e.g. a car that the threshold
  broke into two white patches) are **joined into one object**, and
- the object mask becomes a solid block, which gives a stable bounding box.

Opening is done first (to clean the mask) and dilation afterwards (to make the
objects solid) - in that order, because a dilation followed by an opening would
keep the noise that we want to remove.

## Explanation of Contours

A **contour** is a curve that joins all the continuous points of the same colour
or intensity - here, the outline of every white blob in the mask. Contours are
the natural way to convert a binary mask into object locations.

```python
contours, hierarchy = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

| Argument | Meaning |
| --- | --- |
| `RETR_EXTERNAL` | Return only the outermost contours. Objects inside an object (holes) are ignored, so one vehicle = one contour. |
| `CHAIN_APPROX_SIMPLE` | Store only the corner points of the contour instead of every boundary pixel (saves memory). |

Then, for every contour:

- `cv2.contourArea(contour)` gives the area (in pixels) of the blob. Contours
  with **area <= 500** are ignored as noise; the remaining ones are counted as
  objects.
- `cv2.boundingRect(contour)` gives the smallest upright rectangle
  `(x, y, w, h)` around the contour, which is drawn in green with
  `cv2.rectangle`.
- `cv2.putText` writes the object number just above each box.

## Result / Output

When the script is run, the window **Traffic Object Detection** shows the traffic
video with:

- a **green bounding box** around every detected object,
- the **object number** (1, 2, 3, ...) written above each box,
- `Objects: <n>` in the top-left corner (number of objects in the current frame),
- `Frame: <n>` below it (current frame number).

After the last frame (or after pressing **Q**) the video and the windows are
released and the results are printed in the console:

```
Traffic Object Detection using Background Subtraction (MOG2)
Video  : traffic.mp4
Frames : 1248
FPS    : 30.0
Press Q in the window to stop.


================================
            RESULTS
================================
Total Frames      : 1248
Average Objects   : 12.67
Average Time/Frame: 6.04 ms
Processing FPS    : 165.62
================================
```

Observations made while running the experiment:

- For the first few frames the MOG2 model is still **learning the background**,
  so a few false objects can appear; afterwards only the moving vehicles
  (and similar moving regions) are detected.
- The **average number of objects is about 12-13 per frame** for this video.
  The count changes constantly, because vehicles enter and leave the scene, and
  because two vehicles that touch each other are seen as a single object while
  they are close.
- **Total Frames** is the number of frames that were processed - it is smaller
  than 1248 if the video is stopped early with **Q**.
- **Average Time/Frame** is measured with `time.perf_counter()` around the
  read + process part of every frame; the 30 ms playback delay of `waitKey` is
  **not** included. **Processing FPS** is `total frames / total processing
  time`.
- The values depend on the machine. The output above was measured on a normal
  CPU (Python 3.12, OpenCV 5.0, video 960 x 720 resized to 800 x 600). When the
  display window is used, the time per frame is slightly higher (about 12 ms in
  the same test), because drawing and showing the frame also take time - this is
  still faster than the 30 FPS of the video, so the processing runs in
  real time.

## Conclusion

The experiment was completed successfully: the moving objects of the traffic
video were detected and counted using classic OpenCV image processing only - no
deep learning or tracking model was needed.

- **MOG2 background subtraction** separates the moving objects from the static
  background, and it adapts to slow changes of the background automatically.
- **Binary thresholding at 200** removes the shadows (grey, about 127) that MOG2
  marks, and also removes the weak noise pixels.
- **Morphological opening** removes the small noise blobs, and **dilation** joins
  the parts of the same object, so the object mask is clean and solid.
- **Contours with an area filter (area > 500)** convert the mask into objects and
  reject the remaining noise; the bounding box of each contour gives the position
  of the object.
- The measured **processing FPS (about 165 FPS for 800 x 600 frames)** is much
  higher than the 30 FPS of the video, so the algorithm is fast enough for
  real-time traffic monitoring.

**Limitations:** if the camera shakes, if a vehicle stops moving (it is then
learned as background), or if an object has almost the same colour as the
background and moves very slowly, the detection can miss it or split it into
several objects. A minimum-area value and the MOG2 parameters (`history`,
`varThreshold`) have to be tuned for each new video.
