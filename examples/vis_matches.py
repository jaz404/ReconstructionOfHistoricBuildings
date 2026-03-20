import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def compute_sift_matches(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # KNN matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    return kp1, kp2, good


def ransac_filter(kp1, kp2, matches):
    if len(matches) < 8:
        return [], None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Estimate Essential matrix (assuming focal ~1, no calibration)
    E, mask = cv2.findEssentialMat(
        pts1, pts2,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if mask is None:
        return [], None

    inliers = [m for i, m in enumerate(matches) if mask[i]]

    return inliers, mask

def draw_matches(img1, img2, kp1, kp2, matches, max_draw=200):
    matches_to_draw = matches[:max_draw]
    vis = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(18, 10))
    plt.imshow(vis)
    plt.title(f"RANSAC Inlier Matches")
    plt.axis("off")
    plt.show()

def main(img_path1, img_path2):
    img1 = load_image(img_path1)
    img2 = load_image(img_path2)

    kp1, kp2, good_matches = compute_sift_matches(img1, img2)

    # print(f"Total matches after Lowe: {len(good_matches)}")

    inliers, _ = ransac_filter(kp1, kp2, good_matches)

    # print(f"RANSAC inliers: {len(inliers)}")

    draw_matches(img1, img2, kp1, kp2, inliers)


if __name__ == "__main__":
    main("../data/frames/engine1/frame_00001.png", "../data/frames/engine1/frame_00002.png")