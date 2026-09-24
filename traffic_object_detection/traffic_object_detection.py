"""
Experiment: Traffic Object Detection using Background Subtraction
=================================================================
Detects the moving vehicles / objects of a traffic video with OpenCV, using
MOG2 background subtraction.

Steps applied to every frame of traffic.mp4:
    read frame
      -> resize to 800 x 600
      -> MOG2 background subtraction (history=500, varThreshold=50,
         detectShadows=True)
      -> binary threshold at 200      (removes the shadows + weak noise)
      -> morphological opening, 5x5   (removes the small noise blobs)
      -> dilation, 5x5, 2 iterations  (joins the parts of the same object)
      -> findContours + area filter   (area <= 500 is ignored)
      -> green bounding box + object number
      -> display object count and frame number

Press Q in the video window to stop the processing early.
At the end the program prints: total frames, average objects, average
time per frame and processing FPS.

traffic.mp4 is not stored in the repository (it is a video file), so copy it
next to this script before running, e.g. from the repository root:
    copy video_activity_recognition\traffic.mp4 traffic_object_detection\

Run it from this folder, so that traffic.mp4 is in the working directory:
    python traffic_object_detection.py
"""

import sys      # used to exit with an error code
import time     # used to measure the processing time of every frame

import cv2      # OpenCV


# ======================================================================
# Configuration (the values required by the experiment)
# ======================================================================
VIDEO_PATH = "traffic.mp4"          # input video, in the working directory

FRAME_WIDTH = 800                   # every frame is resized to ...
FRAME_HEIGHT = 600                  # ... 800 x 600 pixels

MOG2_HISTORY = 500                  # frames used by MOG2 to learn the background
MOG2_VAR_THRESHOLD = 50             # squared distance threshold of MOG2
MOG2_DETECT_SHADOWS = True          # shadows are marked grey (value 127)

THRESHOLD_VALUE = 200               # binary threshold applied to the MOG2 mask
KERNEL_SIZE = 5                     # 5 x 5 rectangular kernel
DILATE_ITERATIONS = 2               # the dilation is applied two times
MIN_CONTOUR_AREA = 500              # contours smaller than this are noise

WINDOW_NAME = "Traffic Object Detection"
WAIT_KEY_DELAY = 30                 # ms to wait for a key (about 30 FPS playback)

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)                 # BGR -> green (boxes and numbers)
WHITE = (255, 255, 255)             # BGR -> white (text overlay)


# ======================================================================
# 1. Open the video
# ======================================================================
cap = cv2.VideoCapture(VIDEO_PATH)

# Stop immediately if the video cannot be opened
if not cap.isOpened():
    print(f"Error: could not open the video file '{VIDEO_PATH}'.")
    print("Make sure traffic.mp4 is in the current working directory.")
    print("It is not stored in the repository - copy it from "
          "video_activity_recognition/traffic.mp4.")
    sys.exit(1)

print("Traffic Object Detection using Background Subtraction (MOG2)")
print("Video  :", VIDEO_PATH)
print("Frames :", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print("FPS    :", cap.get(cv2.CAP_PROP_FPS))
print("Press Q in the window to stop.\n")


# ======================================================================
# 2. Create the MOG2 background subtractor and the morphological kernel
# ======================================================================
back_sub = cv2.createBackgroundSubtractorMOG2(
    history=MOG2_HISTORY,
    varThreshold=MOG2_VAR_THRESHOLD,
    detectShadows=MOG2_DETECT_SHADOWS,
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))


# ======================================================================
# 3. Counters used for the final statistics
# ======================================================================
frame_number = 0             # number of frames processed
total_objects = 0            # sum of the objects found in all the frames
total_processing_time = 0.0  # total processing time in seconds


# ======================================================================
# 4. Process the video frame by frame
# ======================================================================
while True:
    # The stopwatch is started before reading the frame, so the measured
    # value is the real processing time of this frame. The waitKey delay
    # (the playback delay) is not included in the measurement.
    start_time = time.perf_counter()

    # (a) read the next frame
    success, frame = cap.read()
    if not success:                  # end of the video (or unreadable frame)
        break
    frame_number += 1

    # (b) resize the frame to 800 x 600
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # (c) background subtraction -> grey-scale foreground mask
    #     (moving objects are bright, the static background is black)
    foreground_mask = back_sub.apply(frame)

    # (d) binary threshold at 200: every pixel above 200 becomes pure white
    #     (255) and the pixels below become black (0). The shadows produced
    #     by MOG2 are grey (about 127), so they are removed here.
    _, binary_mask = cv2.threshold(
        foreground_mask, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY
    )

    # (e) morphological opening (erosion followed by dilation with the same
    #     5 x 5 kernel): removes the small white noise dots of the mask
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # (f) dilation (2 iterations): grows the white regions so that the
    #     objects become solid blocks and the parts of one object join
    final_mask = cv2.dilate(opened_mask, kernel, iterations=DILATE_ITERATIONS)

    # (g) find the contours of the white objects of the mask
    #     (OpenCV 4 and 5 return two values: contours, hierarchy)
    contours, _ = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # (h) keep only the big contours and draw a green box + object number
    object_count = 0
    for contour in contours:
        if cv2.contourArea(contour) <= MIN_CONTOUR_AREA:
            continue                 # too small -> noise, ignore it
        object_count += 1
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        cv2.putText(frame, str(object_count), (x, y - 10), FONT, 0.7, GREEN, 2)

    # (i) write the number of objects and the current frame number
    cv2.putText(frame, f"Objects: {object_count}", (20, 40), FONT, 1.0, WHITE, 2)
    cv2.putText(frame, f"Frame: {frame_number}", (20, 80), FONT, 1.0, WHITE, 2)

    # (j) display the result in the window
    cv2.imshow(WINDOW_NAME, frame)

    # statistics of this frame
    total_objects += object_count
    total_processing_time += time.perf_counter() - start_time

    # (k) press Q (or q) to stop the processing
    key = cv2.waitKey(WAIT_KEY_DELAY) & 0xFF
    if key == ord("q") or key == ord("Q"):
        print("Stopped by the user (Q pressed).")
        break


# ======================================================================
# 5. Release the video and close the OpenCV windows
# ======================================================================
cap.release()
cv2.destroyAllWindows()


# ======================================================================
# 6. Calculate and print the final statistics
# ======================================================================
print("\n================================")
print("            RESULTS")
print("================================")
print("Total Frames      :", frame_number)

if frame_number > 0:
    average_objects = total_objects / frame_number
    average_time_ms = (total_processing_time / frame_number) * 1000.0
    processing_fps = frame_number / total_processing_time

    print("Average Objects   :", round(average_objects, 2))
    print("Average Time/Frame:", round(average_time_ms, 2), "ms")
    print("Processing FPS    :", round(processing_fps, 2))
else:
    print("No frame was processed, no statistics available.")

print("================================")
