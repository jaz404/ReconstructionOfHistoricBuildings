#!/usr/bin/env python3
"""
script_2_sift_match.py

Run SIFT feature extraction and feature matching on a folder of images.

Example usage:
    python feature_matching.py \
        --image_dir ../data/frames/pillar_no_blur \
        --out_dir ../data/frames/pillar_no_blur/sift_matches \
        --pair_mode sequential \
        --ratio 0.75 \
        --use_ransac

    python script_2_sift_match.py \
        --image_dir frames_building \
        --out_dir sift_matches \
        --pair_mode window \
        --window_size 3 \
        --ratio 0.75 \
        --use_ransac
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SIFT feature extraction and matching.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing extracted frames.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save match results.")
    parser.add_argument(
        "--pair_mode",
        type=str,
        default="sequential",
        choices=["sequential", "window", "all"],
        help=(
            "sequential: match image i with i+1 only\n"
            "window: match image i with i+1...i+window_size\n"
            "all: match all pairs"
        ),
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=3,
        help="Used only for pair_mode=window.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe's ratio threshold.",
    )
    parser.add_argument(
        "--use_ransac",
        action="store_true",
        help="Apply RANSAC with the Fundamental Matrix after Lowe's ratio filtering.",
    )
    parser.add_argument(
        "--ransac_reproj_threshold",
        type=float,
        default=1.0,
        help="RANSAC reprojection threshold passed to findFundamentalMat.",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=0,
        help="Max SIFT features per image. 0 means default OpenCV behavior.",
    )
    parser.add_argument(
        "--save_viz",
        action="store_true",
        help="Save match visualization images.",
    )
    parser.add_argument(
        "--resize_for_viz_max_width",
        type=int,
        default=1600,
        help="Max width for saved visualization images.",
    )
    return parser.parse_args()


def list_images(image_dir: Path) -> List[Path]:
    images = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in VALID_EXTS]
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")
    return images


def build_pairs(num_images: int, pair_mode: str, window_size: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []

    if pair_mode == "sequential":
        for i in range(num_images - 1):
            pairs.append((i, i + 1))

    elif pair_mode == "window":
        for i in range(num_images):
            for j in range(i + 1, min(num_images, i + 1 + window_size)):
                pairs.append((i, j))

    elif pair_mode == "all":
        pairs = list(combinations(range(num_images), 2))

    else:
        raise ValueError(f"Unsupported pair_mode: {pair_mode}")

    return pairs


def load_gray_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def create_sift(max_features: int):
    if max_features > 0:
        return cv2.SIFT_create(nfeatures=max_features)
    return cv2.SIFT_create()

def resize_for_viz(image_bgr: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if w <= max_width:
        return image_bgr
    scale = max_width / w
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)

def draw_matches(
    img1_gray: np.ndarray,
    kp1,
    img2_gray: np.ndarray,
    kp2,
    matches,
    inlier_mask=None,
    max_width: int = 1600,
) -> np.ndarray:
    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

    if inlier_mask is not None:
        filtered_matches = [m for m, keep in zip(matches, inlier_mask.ravel().tolist()) if keep]
    else:
        filtered_matches = matches

    vis = cv2.drawMatches(
        img1_gray,
        kp1,
        img2_gray,
        kp2,
        filtered_matches,
        None,
        flags=flags,
    )
    return resize_for_viz(vis, max_width)


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir)
    pairs = build_pairs(len(images), args.pair_mode, args.window_size)

    print(f"[INFO] Found {len(images)} images.")
    print(f"[INFO] Matching {len(pairs)} image pairs with mode='{args.pair_mode}'.")

    sift = create_sift(args.max_features)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    keypoints_list = []
    descriptors_list = []

    print("[INFO] Extracting SIFT features...")
    for img_path in images:
        img = load_gray_image(img_path)
        kp, desc = sift.detectAndCompute(img, None)
        keypoints_list.append(kp)
        descriptors_list.append(desc)
        print(f"  {img_path.name}: {len(kp)} keypoints")

    summary_csv = out_dir / "match_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_1",
            "image_2",
            "kp_1",
            "kp_2",
            "raw_knn_pairs",
            "lowe_matches",
            "ransac_inliers",
            "ransac_used",
        ])

        for idx1, idx2 in pairs:
            path1 = images[idx1]
            path2 = images[idx2]

            kp1 = keypoints_list[idx1]
            kp2 = keypoints_list[idx2]
            desc1 = descriptors_list[idx1]
            desc2 = descriptors_list[idx2]

            if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
                writer.writerow([
                    path1.name,
                    path2.name,
                    len(kp1),
                    len(kp2),
                    0,
                    0,
                    0,
                    int(args.use_ransac),
                ])
                continue

            knn_matches = bf.knnMatch(desc1, desc2, k=2)

            good_matches = []
            for pair in knn_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < args.ratio * n.distance:
                    good_matches.append(m)

            ransac_inliers = 0
            inlier_mask = None

            if args.use_ransac and len(good_matches) >= 8:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                F, mask = cv2.findFundamentalMat(
                    pts1,
                    pts2,
                    method=cv2.FM_RANSAC,
                    ransacReprojThreshold=args.ransac_reproj_threshold,
                    confidence=0.99,
                )

                if F is not None and mask is not None:
                    inlier_mask = mask
                    ransac_inliers = int(mask.sum())
                else:
                    ransac_inliers = 0

            writer.writerow([
                path1.name,
                path2.name,
                len(kp1),
                len(kp2),
                len(knn_matches),
                len(good_matches),
                ransac_inliers,
                int(args.use_ransac),
            ])

            print(
                f"[PAIR] {path1.name} <-> {path2.name} | "
                f"kp=({len(kp1)}, {len(kp2)}) | "
                f"lowe={len(good_matches)} | "
                f"ransac_inliers={ransac_inliers if args.use_ransac else 'N/A'}"
            )

            if args.save_viz:
                img1 = load_gray_image(path1)
                img2 = load_gray_image(path2)
                vis = draw_matches(
                    img1,
                    kp1,
                    img2,
                    kp2,
                    good_matches,
                    inlier_mask=inlier_mask,
                    max_width=args.resize_for_viz_max_width,
                )
                vis_name = f"{path1.stem}__{path2.stem}_matches.jpg"
                cv2.imwrite(str(out_dir / vis_name), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"[INFO] Match summary written to: {summary_csv}")
    print(f"[INFO] Output directory: {out_dir}")


if __name__ == "__main__":
    main()