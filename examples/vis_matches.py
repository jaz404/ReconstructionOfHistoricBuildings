import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img  # keep as BGR internally


def compute_sift_matches(img1_bgr, img2_bgr, sift_params=None, ratio_thresh=0.75, min_size=None):
    if sift_params is None:
        sift_params = {}

    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(**sift_params)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # -------- SIZE FILTERING --------
    if min_size is not None:
        idx1 = [i for i, kp in enumerate(kp1) if kp.size >= min_size]
        idx2 = [i for i, kp in enumerate(kp2) if kp.size >= min_size]

        kp1 = [kp1[i] for i in idx1]
        kp2 = [kp2[i] for i in idx2]

        if des1 is not None:
            des1 = des1[idx1]
        if des2 is not None:
            des2 = des2[idx2]
    # --------------------------------

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return kp1, kp2, [], 0

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    return kp1, kp2, good, len(good)

def ransac_filter(kp1, kp2, matches):
    if len(matches) < 8:
        return [], None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None or mask is None:
        return [], None

    mask = mask.ravel().astype(bool)
    inliers = [m for i, m in enumerate(matches) if mask[i]]

    return inliers, mask


def draw_match_image(img1_bgr, img2_bgr, kp1, kp2, matches, max_draw=200):
    matches_to_draw = matches[:max_draw]

    vis = cv2.drawMatches(
        img1_bgr, kp1,
        img2_bgr, kp2,
        # matches_to_draw,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis_rgb


def compare_sift_settings(img_path1, img_path2):
    img1 = load_image(img_path1)
    img2 = load_image(img_path2)

    settings = [
        {"name": "Default",              "nfeatures": 0,    "nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 1.6},
        {"name": "Fewer features",       "nfeatures": 500,  "nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 1.6},
        {"name": "Higher contrast thr",  "nfeatures": 0,    "nOctaveLayers": 3, "contrastThreshold": 0.08, "edgeThreshold": 10, "sigma": 1.6},
        {"name": "More edge responses",    "nfeatures": 0,    "nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 20, "sigma": 1.6},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.ravel()

    for ax, setting in zip(axes, settings):
        sift_params = {
            "nfeatures": setting["nfeatures"],
            "nOctaveLayers": setting["nOctaveLayers"],
            "contrastThreshold": setting["contrastThreshold"],
            "edgeThreshold": setting["edgeThreshold"],
            "sigma": setting["sigma"],
        }

        kp1, kp2, good_matches, num_good = compute_sift_matches(
            img1, img2, sift_params=sift_params,    
            # min_size=15
        )

        inliers, _ = ransac_filter(kp1, kp2, good_matches)
        num_inliers = len(inliers)

        vis = draw_match_image(img1, img2, kp1, kp2, inliers, max_draw=200)

        ax.imshow(vis)
        ax.set_title(
            f'{setting["name"]}\n'
            f'Lowe matches: {num_good} | RANSAC inliers: {num_inliers}',
            fontsize=11
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_sift_settings(
        "../../data/frames/engine1/frame_00001.png",
        "../../data/frames/engine1/frame_00002.png"
    )