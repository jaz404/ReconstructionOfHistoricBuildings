# 3D Reconstruction of Structures

This repository contains an end-to-end photogrammetry pipeline designed to transition unconstrained mobile video into highly accurate 3D representations. It acts as a robust front-end for 3D Gaussian Splatting (3DGS) and Multi-View Stereo (MVS) by overcoming the inherent brittleness of sequential video tracking and automating the pruning of geometric artifacts (the "Floater Problem").

## View the report: [Final Project Paper](assets/CMPUT428_Final_Project_Paper_Reconstruction_of_Structures.pdf)

## Repository Structure
Based on the execution pipeline, the workspace is organized as follows:

```text
.
├── Makefile                # Master execution script
├── simple_preprocess.py    # Frame extraction and blur detection
├── clahe_window.py         # SIFT extraction + KD-Tree temporal matching
├── inject_data.py          # SQLite database generation and injection
├── run_mapping.py          # COLMAP incremental mapping & undistortion
├── clean_sparse.py         # DBSCAN/SOR point cloud pruning
│
├── images/                 # Directory for extracted frames
├── data_pkl/               # Directory for custom feature matches (.pkl)
├── databases/              # Directory for COLMAP SQLite databases (.db)
├── undistorted/            # Raw COLMAP SfM outputs (bowing corrected)
└── cleaned_sparse/         # Final pruned point clouds ready for 3DGS
```

## Installation & Prerequisites
Ensure you have `python3` installed along with the following dependencies:
* `pycolmap`
* `opencv-python` (OpenCV)
* `numpy`
* `scikit-learn` (for DBSCAN)
* `open3d` (for point cloud processing)
* A working installation of **COLMAP** accessible in your system path.

## Usage & Execution
This pipeline is fully orchestrated via the included `Makefile`. 

To run the entire pipeline from a raw `.mp4` video to a pristine sparse model in one command:
```bash
make pre-all NAME=my_video_name FPS=5
```
*(Note: Place `my_video_name.mp4` in the root directory before running).*

### Step-by-Step Execution
You can also trigger individual stages of the pipeline:

1. **Extract and Preprocess Frames:**
   ```bash
   make preprocess NAME=my_dataset FPS=5
   ```
2. **Extract Features and Match (KD-Tree Window):**
   ```bash
   make features NAME=my_dataset WINDOW_SIZE=15 CLAHE=True
   ```
3. **Inject Matches into COLMAP Database:**
   ```bash
   make inject NAME=my_dataset
   ```
4. **Run Incremental Mapping & Undistort:**
   ```bash
   make mapping NAME=my_dataset
   ```
5. **Clean Sparse Geometry (DBSCAN/SOR):**
   ```bash
   make clean_sparse NAME=my_dataset
   ```

## Pipeline Parameters
You can override the default parameters directly in the command line (e.g., `make features RATIO=0.8`).

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `NAME` | `test_run` | The core identifier for the run. Determines I/O paths. |
| `FPS` | `5` | Subsampling extraction rate from the source `.mp4`. |
| `RATIO` | `0.75` | Lowe's Ratio Test threshold for filtering ambiguous feature matches. |
| `MAX_FEATURES` | `0` | Cap on SIFT features extracted per frame (`0` = unlimited). |
| `WINDOW_SIZE` | `15` | The temporal neighborhood ($t \pm 15$) used for the KD-Tree matcher. |
| `CLAHE` | `True` | Applies Contrast Limited Adaptive Histogram Equalization to mitigate the aperture problem on reflective surfaces. |
| `BLUR` | `False` | Toggles programmatic dropping of frames with high Laplacian variance. |
| `POTRAIT` | `0` | Orientation flag. Set to `1` to correct camera intrinsics for `1080x1920` portrait aspect ratios vs. standard landscape. |
| `F` | *(None)* | Optional override to manually pass a focal length parameter to the camera model. |

## Pipeline Output
The final, 3DGS-ready output will be located in:
`cleaned_sparse/[NAME]/`

This folder contains the `cameras.bin`, `images.bin`, and `points3D.bin` files, representing the mathematically pruned point cloud. Feed this directory directly into your Gaussian Splatting engine of choice to bypass densification failures and the "Floater Problem."
