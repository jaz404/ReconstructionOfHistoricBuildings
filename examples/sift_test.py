import cv2
import matplotlib.pyplot as plt

img_path = r"..\..\data\frames\engine1\frame_00000.png"

img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Could not load image: {img_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

settings = [
    {"name": "Default",                "nfeatures": 0,    "nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 1.6},
    {"name": "Fewer features",         "nfeatures": 500,  "nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 1.6},
    {"name": "More retained",          "nfeatures": 1500, "nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 1.6},
    {"name": "Higher contrast thr",    "nfeatures": 0,    "nOctaveLayers": 3, "contrastThreshold": 0.08, "edgeThreshold": 10, "sigma": 1.6},
    {"name": "Lower contrast thr",     "nfeatures": 0,    "nOctaveLayers": 3, "contrastThreshold": 0.02, "edgeThreshold": 10, "sigma": 1.6},
    {"name": "Stricter edge filter",   "nfeatures": 0,    "nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 5,  "sigma": 1.6},
    {"name": "More edge responses",    "nfeatures": 0,    "nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 15, "sigma": 1.6},
    {"name": "More smoothing",         "nfeatures": 0,    "nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 2.0},
]

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
axes = axes.ravel()

for ax, s in zip(axes, settings):
    sift = cv2.SIFT_create(
        nfeatures=s["nfeatures"],
        nOctaveLayers=s["nOctaveLayers"],
        contrastThreshold=s["contrastThreshold"],
        edgeThreshold=s["edgeThreshold"],
        sigma=s["sigma"]
    )

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    min_size = 10  # try values like 8, 10, 12, 15

    filtered_kp = [kp for kp in keypoints if kp.size >= min_size]
    
    out = cv2.drawKeypoints(
        img,
        filtered_kp,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    ax.imshow(out_rgb)
    ax.set_title(f'{s["name"]}\n{len(keypoints)} keypoints', fontsize=11)
    ax.axis("off")

plt.tight_layout()
plt.show()