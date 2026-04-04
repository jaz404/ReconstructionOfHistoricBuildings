import argparse
import pycolmap
import open3d as o3d
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Clean a COLMAP sparse model by removing outliers.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input COLMAP sparse model directory (e.g., undistorted/NAME/0/sparse).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the cleaned COLMAP model.")
    return parser.parse_args()

def clean_bin(args=None):
    if args is None:
        args = parse_args()
    # 1. Define Paths
    # Point this to the sparse folder INSIDE your perfect_dataset
    input_model_dir = Path(args.input_dir)
    output_model_dir = Path(args.output_dir)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Loading perfectly flat COLMAP model from {input_model_dir}...")
    recon = pycolmap.Reconstruction(input_model_dir)

    # 2. Extract points and their true COLMAP IDs into parallel arrays
    point_ids = []
    xyz = []
    colors = [] 

    for p_id, point in recon.points3D.items():
        point_ids.append(p_id)
        xyz.append(point.xyz)
        colors.append(point.color / 255.0) # Open3D expects colors between 0 and 1

    point_ids = np.array(point_ids)
    xyz = np.array(xyz)
    colors = np.array(colors)

    print(f"  -> Extracted {len(point_ids)} points.")

    # 3. Feed the coordinates to Open3D 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ==========================================
    # FILTER 1: Statistical Outlier Removal
    # ==========================================
    print("\n[INFO] Running Statistical Outlier Removal...")
    cl, sor_indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_clean = pcd.select_by_index(sor_indices)

    # CRITICAL: We must filter our ID array so it perfectly matches Open3D's surviving points
    point_ids_clean = point_ids[sor_indices]
    print(f"  -> Points surviving SOR: {len(point_ids_clean)}")

    # ==========================================
    # FILTER 2: DBSCAN Clustering (Isolate the Building)
    # ==========================================
    print("\n[INFO] Running DBSCAN Clustering...")
    labels = np.array(pcd_clean.cluster_dbscan(eps=0.5, min_points=10, print_progress=False))

    if len(labels) > 0:
        largest_cluster_idx = np.bincount(labels[labels >= 0]).argmax()
        dbscan_indices = np.where(labels == largest_cluster_idx)[0]
        
        # These are the final, mathematically verified COLMAP point IDs we want to keep
        surviving_point_ids = set(point_ids_clean[dbscan_indices])
        print(f"  -> Kept main building cluster. Points surviving DBSCAN: {len(surviving_point_ids)}")
    else:
        print("[WARNING] Clustering failed. Falling back to SOR clean cloud.")
        surviving_point_ids = set(point_ids_clean)

    # ==========================================
    # EXECUTION: Delete the noise from COLMAP
    # ==========================================
    all_point_ids = set(point_ids)
    points_to_delete = all_point_ids - surviving_point_ids

    print(f"\n[INFO] Surgically deleting {len(points_to_delete)} floaters from the 3D model...")

    # Delete the points from the PyCOLMAP Reconstruction object
    for p_id in points_to_delete:
        recon.delete_point3D(p_id)

    # Save the new, flawless .bin files
    recon.write(output_model_dir)

    print(f"\n[SUCCESS] Clean model saved!")
    print(f"Point your 3DGS train.py to: {output_model_dir}")


def verify(args):
    # 1. Point this to the NEW, cleaned folder we just generated
    clean_model_dir = Path(args.output_dir) # This should be the same as the output_model_dir from clean_bin()
    verification_ply = Path(args.output_dir) / "verification_cleaned.ply"

    print(f"[INFO] Reading the modified COLMAP database...")
    try:
        # Load the cleaned geometry
        clean_recon = pycolmap.Reconstruction(clean_model_dir)
        
        # Print the math
        print(f"  -> Cameras intact: {clean_recon.num_reg_images()}")
        print(f"  -> Points remaining: {clean_recon.num_points3D()}")
        
        # Export it for your human eyes
        clean_recon.export_PLY(str(verification_ply))
        print(f"\n[SUCCESS] Verification PLY saved to: {verification_ply}")
        
    except Exception as e:
        print(f"[ERROR] Failed to read the cleaned model: {e}")

def main():
    args = parse_args()
    clean_bin(args)
    verify(args)

if __name__ == "__main__":
    main()