import pycolmap
from pathlib import Path

def main():
    db_path = "databases/colmap_data_max.db"
    image_dir = "images/engine1"
    output_dir = Path("sparse/0")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create the MASTER pipeline options object
    # This is what the 'incremental_mapping' function signature is asking for
    pipeline_options = pycolmap.IncrementalPipelineOptions()

    # 2. Access the 'mapper' attribute inside the pipeline options
    # We apply your "Sledgehammer" fixes here
    pipeline_options.mapper.init_min_tri_angle = 0.5
    pipeline_options.mapper.init_max_error = 12.0
    pipeline_options.mapper.abs_pose_max_error = 12.0 # Matches your error log attempt
    
    print(f"\n[INFO] Starting PyCOLMAP with correct PipelineOptions structure...")
    
    try:
        # Pass the pipeline_options (not just mapper_options)
        reconstructions = pycolmap.incremental_mapping(
            database_path=db_path,
            image_path=image_dir,
            output_path=output_dir,
            options=pipeline_options  # This now matches the expected type
        )
        
        if reconstructions:
            print(f"\n[SUCCESS] Sparse 3D Point Cloud built!")
            for rec_id, rec in reconstructions.items():
                print(f"  -> Model {rec_id}: {rec.num_reg_images()} images, {rec.num_points3D()} points.")
        else:
            print("\n[FAILED] Failed to create any sparse model. Check image overlap/parallax.")

    except Exception as e:
        print(f"\n[ERROR] PyCOLMAP Exception: {e}")

    # Assuming 'reconstruction' is the object from Model 1
    # This usually lives in reconstructions[1] based on your log
    if reconstructions:
        model_0 = reconstructions[0] 
        
        # Export to PLY for 3dviewer.net
        model_0.export_PLY("sparse/0/0/model.ply")
        print("[SUCCESS] Exported model.ply to sparse/0/0/")

if __name__ == "__main__":
    main()