import time
from pathlib import Path
import cv2


def display_video_information(video_path: str) -> None:
    """
    Reads a video file, displays its properties and plays it
    frame-by-frame using OpenCV.

    Controls:
        Q or Esc : Stop the video
        Space    : Pause or resume
    """

    # Remove accidental quotation marks from the path
    video_path = video_path.strip().strip('"').strip("'")
    path = Path(video_path).expanduser()

    # Check whether the file exists
    if not path.is_file():
        print(f"Error: Video file not found: {path}")
        return

    # Open the video file
    video = cv2.VideoCapture(str(path))

    if not video.isOpened():
        print("Error: OpenCV could not open the video file.")
        return

    # Extract video properties
    video_name = path.name

    frame_width = int(
        video.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        video.get(cv2.CAP_PROP_FPS)
    )

    # Calculate duration
    if fps > 0:
        duration_seconds = total_frames / fps
    else:
        duration_seconds = 0

    duration_minutes = int(duration_seconds // 60)
    remaining_seconds = duration_seconds % 60

    # Display video properties
    print("\n========== VIDEO PROPERTIES ==========")
    print(f"Video Name       : {video_name}")
    print(
        f"Resolution       : "
        f"{frame_width} × {frame_height} pixels"
    )
    print(f"Number of Frames : {total_frames}")
    print(f"Frame Rate       : {fps:.2f} FPS")
    print(f"Duration         : {duration_seconds:.2f} seconds")
    print(
        f"Duration Format  : "
        f"{duration_minutes:02d}:{remaining_seconds:05.2f}"
    )
    print("======================================")

    # Calculate delay between frames in milliseconds
    if fps > 0:
        frame_delay = max(1, round(1000 / fps))
    else:
        frame_delay = 30

    frame_number = 0
    paused = False

    # Real-time FPS calculation variables
    fps_update_interval = 0.5          # update FPS display every 0.5 sec
    prev_tick = time.perf_counter()
    fps_counter = 0
    current_fps = 0.0

    print("\nControls")
    print("Press SPACE to pause or resume.")
    print("Press Q or ESC to stop.")

    try:
        while True:

            if not paused:
                success, frame = video.read()

                # Stop after the final frame
                if not success:
                    print("\nEnd of video reached.")
                    break

                frame_number += 1
                fps_counter += 1

                # Calculate real-time FPS every fps_update_interval seconds
                now = time.perf_counter()
                elapsed = now - prev_tick
                if elapsed >= fps_update_interval:
                    current_fps = fps_counter / elapsed
                    fps_counter = 0
                    prev_tick = now

                # Add frame information to the displayed video
                frame_label = (
                    f"Frame: {frame_number}/{total_frames}"
                )

                cv2.putText(
                    frame,
                    frame_label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                # Overlay real-time FPS on the video
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(
                    frame,
                    fps_text,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow(
                    "AI-Driven Video Processing",
                    frame
                )

            # Keep the window responsive when paused
            delay = 30 if paused else frame_delay
            key = cv2.waitKey(delay) & 0xFF

            # Q or Esc stops the video
            if key == ord("q") or key == 27:
                print("\nVideo stopped by the user.")
                break

            # Space pauses or resumes the video
            if key == ord(" "):
                paused = not paused

    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    file_path = input(
        "Enter the complete path of the video file: "
    )

    display_video_information(file_path)