import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img


def create_extractor(name):
    if name == "SIFT":
        return cv2.SIFT_create(), cv2.NORM_L2
    elif name == "ORB":
        return cv2.ORB_create(nfeatures=1500), cv2.NORM_HAMMING
    elif name == "AKAZE":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    elif name == "BRISK":
        return cv2.BRISK_create(), cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unknown extractor: {name}")


def compute_matches(img1, img2, extractor_name, ratio_thresh=0.75):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    extractor, norm_type = create_extractor(extractor_name)

    kp1, des1 = extractor.detectAndCompute(gray1, None)
    kp2, des2 = extractor.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return kp1, kp2, []

    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    return kp1, kp2, good


def ransac_filter(kp1, kp2, matches):
    if len(matches) < 8:
        return []

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(
        pts1, pts2,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None or mask is None:
        return []

    mask = mask.ravel().astype(bool)
    return [m for i, m in enumerate(matches) if mask[i]]


def draw_matches(img1, img2, kp1, kp2, matches, max_draw=200):
    vis = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        # matches[:max_draw],
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def compare_extractors(img_path1, img_path2):
    img1 = load_image(img_path1)
    img2 = load_image(img_path2)

    extractors = ["SIFT", "ORB", "AKAZE", "BRISK"]

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.ravel()

    for ax, name in zip(axes, extractors):
        kp1, kp2, good = compute_matches(img1, img2, name)
        inliers = ransac_filter(kp1, kp2, good)
        vis = draw_matches(img1, img2, kp1, kp2, inliers)

        ax.imshow(vis)
        ax.set_title(f"{name}\nLowe: {len(good)} | RANSAC: {len(inliers)}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_extractors(
        "../../data/frames/engine1/frame_00001.png",
        "../../data/frames/engine1/frame_00002.png"
    )