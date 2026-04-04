# %%writefile feature_extraction.py
import argparse
import pickle
from pathlib import Path
import cv2
import numpy as np

def preprocess(gray, use_clahe=False, use_blur=False):
    out = gray.copy()

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        out = clahe.apply(out)

    if use_blur:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    return out

def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter for image files
    images = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    
    # 1. Initialize Standard SIFT (CPU)
    # This works everywhere and is very stable
    # detector = cv2.SIFT_create(nfeatures=args.max_features if args.max_features > 0 else 0)
    
    detector = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.01,
        edgeThreshold=20,
        sigma=1.6
    )


    matcher = cv2.BFMatcher(cv2.NORM_L2)

    export_data = {"keypoints": {}, "matches": {}}
    descriptors_list = []
    
    print(f"[INFO] Extracting features for {len(images)} images...")

    for img_path in images:
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        gray = preprocess(img, use_clahe=args.clahe, use_blur=args.blur)
        if img is None:
            continue

        # Detect and Compute
        kp, desc = detector.detectAndCompute(gray, None)
        
        # Save Keypoints for COLMAP
        kp_array = np.float32([k.pt for k in kp]) if kp else np.empty((0, 2), dtype=np.float32)
        export_data["keypoints"][img_path.name] = kp_array
        descriptors_list.append(desc)
        print(f"  {img_path.name}: {len(kp)} keypoints found")

    # 2. Sequential Matching Loop
    print("[INFO] Matching pairs...")
    for i in range(len(images) - 1):
        idx1, idx2 = i, i + 1
        name1, name2 = images[idx1].name, images[idx2].name
        
        desc1, desc2 = descriptors_list[idx1], descriptors_list[idx2]
        if desc1 is None or desc2 is None:
            continue

        # KNN Match
        knn_matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Lowe's Ratio Test
        good_matches = []
        for m, n in knn_matches:
            if m.distance < args.ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) >= 8:
            # Prepare points for RANSAC Fundamental Matrix
            pts1 = export_data["keypoints"][name1][[m.queryIdx for m in good_matches]]
            pts2 = export_data["keypoints"][name2][[m.trainIdx for m in good_matches]]

            # Find Fundamental Matrix
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)

            if F is not None and mask is not None:
                mask_flat = mask.ravel()
                # Only keep the indices that passed RANSAC
                valid_indices = [[good_matches[j].queryIdx, good_matches[j].trainIdx] 
                                for j in range(len(good_matches)) if mask_flat[j] == 1]
                
                if len(valid_indices) > 0:
                    export_data["matches"][f"{name1}|{name2}"] = {
                        "indices": np.array(valid_indices, dtype=np.uint32),
                        "F": F.astype(np.float64)
                    }

    # 3. Export the .pkl file
    pkl_path = out_dir / "csc_data_clahe.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(export_data, f)
    
    print(f"\n[SUCCESS] Exported {pkl_path}")
    print(f"[INFO] Run inject_data.py on your Mac.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=0.75)
    # parser.add_argument("--max_features", type=int, default=2000)
    parser.add_argument("--clahe", default=True)
    parser.add_argument("--blur", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    main()