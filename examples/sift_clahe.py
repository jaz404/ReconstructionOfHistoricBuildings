import cv2
import matplotlib.pyplot as plt

img_path = r"..\..\data\frames\csc_frontface\frame_00072.png"
FILTER_KP = False
FILTER_MODE = "response"   # "size" or "response"
MIN_SIZE = 10
TOP_K = 800

img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Could not load image: {img_path}")

gray_base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

settings = [
    {
        "name": "Default",
        "nfeatures": 0, "nOctaveLayers": 3, "contrastThreshold": 0.04,
        "edgeThreshold": 10, "sigma": 1.6,
        "clahe": False, "blur": False
    },
    {
        "name": "Fewer features",
        "nfeatures": 500, "nOctaveLayers": 3, "contrastThreshold": 0.04,
        "edgeThreshold": 10, "sigma": 1.6,
        "clahe": False, "blur": False
    },
    {
        "name": "More retained",
        "nfeatures": 1500, "nOctaveLayers": 3, "contrastThreshold": 0.04,
        "edgeThreshold": 10, "sigma": 1.6,
        "clahe": False, "blur": False
    },
    {
        "name": "Higher contrast thr",
        "nfeatures": 0, "nOctaveLayers": 3, "contrastThreshold": 0.08,
        "edgeThreshold": 10, "sigma": 1.6,
        "clahe": False, "blur": False
    },
    {
        "name": "Lower contrast thr",
        "nfeatures": 0, "nOctaveLayers": 3, "contrastThreshold": 0.02,
        "edgeThreshold": 10, "sigma": 1.6,
        "clahe": False, "blur": False
    },
    {
        "name": "Stricter edge filter",
        "nfeatures": 0, "nOctaveLayers": 3, "contrastThreshold": 0.04,
        "edgeThreshold": 5, "sigma": 1.6,
        "clahe": False, "blur": False
    },
    {
        "name": "More edge responses",
        "nfeatures": 0, "nOctaveLayers": 3, "contrastThreshold": 0.04,
        "edgeThreshold": 15, "sigma": 1.6,
        "clahe": False, "blur": False
    },
    {
        "name": "edge + lower contrast",
        "nfeatures": 0, "nOctaveLayers": 3, "contrastThreshold": 0.02,
        "edgeThreshold": 25, "sigma": 1.6,
        "clahe": False, "blur": False
    },

    # Added building-friendly variants
    {
        "name": "CLAHE + low contrast",
        "nfeatures": 0, "nOctaveLayers": 3, "contrastThreshold": 0.02,
        "edgeThreshold": 10, "sigma": 1.6,
        "clahe": True, "blur": False
    },
    {
        "name": "CLAHE + very low contrast",
        "nfeatures": 0, "nOctaveLayers": 3, "contrastThreshold": 0.01,
        "edgeThreshold": 10, "sigma": 1.6,
        "clahe": True, "blur": False
    },
    {
        "name": "CLAHE + edge + low contrast",
        "nfeatures": 0, "nOctaveLayers": 3, "contrastThreshold": 0.01,
        "edgeThreshold": 20, "sigma": 1.6,
        "clahe": True, "blur": False
    },
    {
        "name": "CLAHE + blur + edge",
        "nfeatures": 0, "nOctaveLayers": 3, "contrastThreshold": 0.01,
        "edgeThreshold": 20, "sigma": 1.6,
        "clahe": True, "blur": True
    },
]


def preprocess(gray, use_clahe=False, use_blur=False):
    out = gray.copy()

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        out = clahe.apply(out)

    if use_blur:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    return out


fig, axes = plt.subplots(3, 4, figsize=(22, 16))
axes = axes.ravel()

for ax, s in zip(axes, settings):
    gray = preprocess(gray_base, use_clahe=s["clahe"], use_blur=s["blur"])

    sift = cv2.SIFT_create(
        nfeatures=s["nfeatures"],
        nOctaveLayers=s["nOctaveLayers"],
        contrastThreshold=s["contrastThreshold"],
        edgeThreshold=s["edgeThreshold"],
        sigma=s["sigma"]
    )

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if FILTER_KP:
        if FILTER_MODE == "size":
            keypoints = [kp for kp in keypoints if kp.size >= MIN_SIZE]
        elif FILTER_MODE == "response":
            keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:TOP_K]

    out = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    extra = []
    if s["clahe"]:
        extra.append("CLAHE")
    if s["blur"]:
        extra.append("Blur")

    subtitle = f'\n{", ".join(extra)}' if extra else ""
    ax.imshow(out_rgb)
    ax.set_title(f'{s["name"]}{subtitle}\n{len(keypoints)} keypoints', fontsize=11)
    ax.axis("off")

# Hide unused axes if any
for i in range(len(settings), len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.show()