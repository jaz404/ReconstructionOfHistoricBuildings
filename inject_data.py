import pickle
import numpy as np
from pathlib import Path
from database import COLMAPDatabase

def main():
    db_path = "databases/csc_front.db"
    pickle_path = "data_pkl/csc_front.pkl"
    
    # Check if data exists
    if not Path(pickle_path).exists():
        print(f"[ERROR] Could not find {pickle_path}")
        return

    # Initialize the empty database
    # If a database already exists here, COLMAPDatabase will append to it. 
    # To start fresh, delete the old one.
    if Path(db_path).exists():
        inp = input(f"[WARNING] {db_path} already exists. Do you want to delete it and start fresh? (y/n): ")
        if inp.lower() == "y":
            Path(db_path).unlink()
            print(f"[INFO] Deleted old {db_path} to start fresh.")
        else:
            print("[INFO] Exiting to avoid duplicate injection.")
            return

    db = COLMAPDatabase(db_path)
    print(f"[INFO] Created new COLMAP database at {db_path}")
    
    # CAMERA MODEL FIX: 
    # Model 1 = PINHOLE (Required for Gaussian Splatting)
    # radially corrected the mobile cam.(2)
    # TODO: Update width/height and focal length params

    width, height = 1920, 1080
    focal_length = 1500.0  # in pixels
    cx, cy = width / 2, height / 2

    cam_id = db.add_camera(model=1, width=width, height=height, params=[focal_length, focal_length, cx, cy])
    print(f"[INFO] Registered Camera ID: {cam_id}")

    print(f"[INFO] Loading feature data from {pickle_path}...")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # Inject Images and Keypoints
    image_id_map = {} # Maps "frame_001.jpg" -> Database ID (e.g., 1)

    print("[INFO] Injecting Images and Keypoints...")
    for filename, keypoints in data["keypoints"].items():
        # Register the image in the DB and get its unique integer ID
        img_id = db.add_image(filename, cam_id)
        image_id_map[filename] = img_id
        
        # Inject the Nx2 numpy array of X,Y coordinates
        db.add_keypoints(img_id, keypoints)
        print(f"  -> Added {filename} (ID: {img_id}) with {len(keypoints)} keypoints.")

    # Inject Verified Matches and Fundamental Matrices
    print("[INFO] Injecting Two-View Geometries (Matches)...")
    match_count = 0
    for pair_name, match_info in data["matches"].items():
        # pair_name looks like "frame_001.jpg|frame_002.jpg"
        img1_name, img2_name = pair_name.split("|")
        
        # Look up their integer IDs
        img1_id = image_id_map[img1_name]
        img2_id = image_id_map[img2_name]
        
        # Extract the arrays
        indices = match_info["indices"]
        f_matrix = match_info["F"]
        
        # Inject into the database
        db.add_matches(img1_id, img2_id, indices, F_matrix=f_matrix)
        match_count += 1
        
    print(f"[INFO] Injected {match_count} verified image pairs.")

    # Save and close
    db.commit()
    print("[SUCCESS] Database injection complete! Ready for COLMAP mapper.")

if __name__ == "__main__":
    main()