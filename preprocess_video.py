#!/usr/bin/env python3
"""
preprocess_video.py

Extract frames from a video 

Example usage:
    python preprocess_video.py \
        --video ../data/pillar.mp4 \
        --out_dir ../data/frames/pillar \
        --target_fps 2 \
        --max_width 1920 \
        --blur_threshold 120.0

    python preprocess_video.py \
        --video building.mp4 \
        --out_dir frames_building \
        --frame_stride 15 \
        --scale 0.25 \
        --keep_blurry
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and preprocess frames from video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save extracted frames.")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--target_fps",
        type=float,
        default=None,
        help="Desired extraction FPS. Example: 2.0 means save ~2 frames per second.",
    )
    group.add_argument(
        "--frame_stride",
        type=int,
        default=None,
        help="Save every Nth frame. Example: 15 means save every 15th frame.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Uniform resize scale factor. Example: 0.25 for 8K->2K-ish.",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=None,
        help="Resize so output width does not exceed this value while preserving aspect ratio.",
    )
    parser.add_argument(
        "--max_height",
        type=int,
        default=None,
        help="Resize so output height does not exceed this value while preserving aspect ratio.",
    )

    parser.add_argument(
        "--blur_threshold",
        type=float,
        default=None,
        help=(
            "Variance of Laplacian threshold for blur filtering. "
            "Frames below this are discarded unless --keep_blurry is used. "
            "Good starting range: 80 to 200."
        ),
    )
    parser.add_argument(
        "--keep_blurry",
        action="store_true",
        help="Keep blurry frames but still record blur score in metadata.",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image format.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality if saving jpg/jpeg.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def variance_of_laplacian(image_bgr) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_resize_shape(
    width: int,
    height: int,
    scale: Optional[float],
    max_width: Optional[int],
    max_height: Optional[int],
) -> Tuple[int, int]:
    new_w, new_h = width, height

    if scale is not None:
        if scale <= 0:
            raise ValueError("--scale must be > 0.")
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))

    if max_width is not None or max_height is not None:
        width_scale = 1.0
        height_scale = 1.0

        if max_width is not None and new_w > max_width:
            width_scale = max_width / new_w
        if max_height is not None and new_h > max_height:
            height_scale = max_height / new_h

        limiting_scale = min(width_scale, height_scale)
        if limiting_scale < 1.0:
            new_w = max(1, int(round(new_w * limiting_scale)))
            new_h = max(1, int(round(new_h * limiting_scale)))

    return new_w, new_h


def should_save_frame(frame_idx: int, video_fps: float, target_fps: Optional[float], frame_stride: Optional[int]) -> bool:
    if frame_stride is not None:
        return frame_idx % frame_stride == 0

    if target_fps is not None:
        if target_fps <= 0:
            raise ValueError("--target_fps must be > 0.")
        step = max(1, int(round(video_fps / target_fps)))
        return frame_idx % step == 0

    return True


def save_image(path: Path, image_bgr, ext: str, jpeg_quality: int) -> None:
    if ext in {"jpg", "jpeg"}:
        ok = cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    else:
        ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise IOError(f"Failed to save image to {path}")


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_w, new_h = compute_resize_shape(
        orig_width,
        orig_height,
        args.scale,
        args.max_width,
        args.max_height,
    )

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Original resolution: {orig_width}x{orig_height}")
    print(f"[INFO] Output resolution:   {new_w}x{new_h}")
    print(f"[INFO] Video FPS: {video_fps:.3f}")
    print(f"[INFO] Total frames: {frame_count}")

    metadata_path = out_dir / "frame_metadata.csv"
    saved_count = 0
    examined_count = 0
    discarded_blurry = 0

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_video",
            "frame_index",
            "timestamp_sec",
            "saved_filename",
            "saved",
            "blur_score",
            "discarded_for_blur",
            "orig_width",
            "orig_height",
            "out_width",
            "out_height",
        ])

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if not should_save_frame(frame_idx, video_fps, args.target_fps, args.frame_stride):
                frame_idx += 1
                continue

            examined_count += 1

            if (new_w, new_h) != (orig_width, orig_height):
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            blur_score = variance_of_laplacian(frame)
            is_blurry = args.blur_threshold is not None and blur_score < args.blur_threshold

            timestamp_sec = frame_idx / video_fps if video_fps > 0 else 0.0
            filename = f"{args.prefix}_{saved_count:05d}.{args.image_ext}"

            save_this = True
            if is_blurry and not args.keep_blurry:
                save_this = False
                discarded_blurry += 1
                filename = ""

            if save_this:
                save_path = out_dir / filename
                save_image(save_path, frame, args.image_ext, args.jpeg_quality)
                saved_count += 1

            writer.writerow([
                str(video_path),
                frame_idx,
                f"{timestamp_sec:.6f}",
                filename,
                int(save_this),
                f"{blur_score:.6f}",
                int(is_blurry),
                orig_width,
                orig_height,
                new_w,
                new_h,
            ])

            frame_idx += 1

    cap.release()

    print(f"[INFO] Examined candidate frames: {examined_count}")
    print(f"[INFO] Saved frames: {saved_count}")
    print(f"[INFO] Discarded blurry frames: {discarded_blurry}")
    print(f"[INFO] Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()