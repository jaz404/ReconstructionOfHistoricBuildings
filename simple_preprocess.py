import cv2
import argparse
from pathlib import Path

def extract_frames(video_path, out_dir, target_fps=1.5, max_dimension=1920):
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Video loaded: {video_path.name}")
    print(f"[INFO] Original Specs: {orig_w}x{orig_h} at {original_fps:.2f} FPS")
    
    frame_interval = int(round(original_fps / target_fps))
    print(f"[INFO] Extracting 1 frame every {frame_interval} frames (Target: {target_fps} FPS)")

    # Check if scaling is actually needed
    needs_scaling = max(orig_h, orig_w) > max_dimension
    
    if needs_scaling:
        # Calculate new dimensions (Scaling down to max_dimension safely)
        scale = max_dimension / max(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        print(f"[INFO] Downscaling frames to: {new_w}x{new_h} to save memory")
    else:
        print(f"[INFO] Video resolution is below the {max_dimension} limit. Skipping scaling.")

    current_frame = 314
    saved_count = 314

    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        if current_frame % frame_interval == 0:
            # Only apply resize if the video is larger than the max_dimension
            if needs_scaling:
                frame_to_save = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame_to_save = frame
            
            # Save with zero-padding (image_0001.jpg)
            out_filename = out_dir / f"image_{saved_count:04d}.jpg"
            cv2.imwrite(str(out_filename), frame_to_save)
            saved_count += 1

        current_frame += 1

    cap.release()
    print(f"[SUCCESS] Extracted {saved_count} optimized frames to {out_dir}/")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract and downscale frames from a video.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--out_dir", type=str, default="images", help="Directory to save extracted frames")
    parser.add_argument("--fps", type=float, default=1.5, help="Target frames per second to extract")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    extract_frames(args.video, args.out_dir, args.fps)