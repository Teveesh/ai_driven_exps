import cv2
import os
from pathlib import Path

def extract_frames(video_path: str, output_dir: str = "extracted_frames", 
                   frame_interval: int = 1, start_frame: int = 0, 
                   end_frame: int = None, extract_all: bool = True):
    """
    Extract frames from a video file and save them as images.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (default: 1 = all frames)
        start_frame: Starting frame index (0-based)
        end_frame: Ending frame index (None = until end)
        extract_all: If True, extracts all frames. If False, uses frame_interval.
    """
    video_path = video_path.strip().strip('"').strip("'")
    path = Path(video_path).expanduser()
    
    if not path.is_file():
        print(f"Error: Video file not found: {path}")
        return
    
    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Open video
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        print("Error: Could not open video file.")
        return
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {path.name}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Output directory: {out_path.resolve()}")
    print()
    
    if end_frame is None:
        end_frame = total_frames
    
    frame_number = 0
    saved_count = 0
    skip_count = 0
    
    while True:
        success, frame = video.read()
        if not success:
            break
        
        # Check if we're past the end frame
        if frame_number >= end_frame:
            break
        
        # Check if we're past the start frame
        if frame_number >= start_frame:
            # Decide whether to save this frame
            should_save = extract_all or ((frame_number - start_frame) % frame_interval == 0)
            
            if should_save:
                # Create filename with zero-padded frame number
                filename = f"frame_{frame_number:06d}.jpg"
                filepath = out_path / filename
                cv2.imwrite(str(filepath), frame)
                saved_count += 1
                
                # Progress indicator
                if saved_count % 50 == 0:
                    print(f"  Saved {saved_count} frames...")
            else:
                skip_count += 1
        
        frame_number += 1
    
    video.release()
    
    print(f"\nExtraction complete!")
    print(f"  Frames processed: {frame_number}")
    print(f"  Frames saved: {saved_count}")
    print(f"  Frames skipped: {skip_count}")
    print(f"  Output: {out_path.resolve()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video_path", nargs="?", 
                        default=r"C:\Users\tevee\Downloads\From Klickpin.com- 89 Trending Summer Salad Recipes for This Year-pin-id-296674694227142129.mp4",
                        help="Path to the video file")
    parser.add_argument("--output", "-o", default="extracted_frames",
                        help="Output directory for frames")
    parser.add_argument("--interval", "-i", type=int, default=1,
                        help="Extract every Nth frame (default: 1)")
    parser.add_argument("--start", "-s", type=int, default=0,
                        help="Starting frame index (default: 0)")
    parser.add_argument("--end", "-e", type=int, default=None,
                        help="Ending frame index (default: end of video)")
    parser.add_argument("--skip", action="store_true",
                        help="Use interval-based extraction instead of all frames")
    
    args = parser.parse_args()
    
    extract_frames(
        video_path=args.video_path,
        output_dir=args.output,
        frame_interval=args.interval,
        start_frame=args.start,
        end_frame=args.end,
        extract_all=not args.skip
    )