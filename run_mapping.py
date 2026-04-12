import argparse

import pycolmap
from pathlib import Path

def main():
    args = parse_args()

    db_path = args.db
    image_dir = args.image_dir
    output_dir = Path(args.output_dir)
    undistorted_dir = Path(args.undistorted_dir)
    
    # 1. Create the MASTER pipeline options object
    # This is what the 'incremental_mapping' function signature is asking for
    pipeline_options = pycolmap.IncrementalPipelineOptions()

    # 2. Access the 'mapper' attribute inside the pipeline options
    # pipeline_options.mapper.init_min_tri_angle = 2.0
    # pipeline_options.mapper.init_max_error = 8.0
    # pipeline_options.mapper.abs_pose_max_error = 8.0 
    pipeline_options.mapper.filter_max_reproj_error = 3.0   # slightly tighter reprojection error filter.
    
    print(f"\n[INFO] Starting PyCOLMAP with correct PipelineOptions structure...")

    try:
        # 1. Run the Incremental Mapper
        reconstructions = pycolmap.incremental_mapping(
            database_path=db_path,
            image_path=image_dir,
            output_path=output_dir,
            options=pipeline_options
        )
        
        if reconstructions:
            print(f"\n[SUCCESS] Sparse 3D Point Cloud(s) built!")
            print(f"\n[INFO] Starting PyCOLMAP API Undistorter on ALL generated models...")
            
            # 2. Loop through every model COLMAP created
            for rec_id, model in reconstructions.items():
                print(f"\n--- Processing Model {rec_id} ({model.num_reg_images()} images, {model.num_points3D()} points) ---")
                
                # Setup Directories
                model_sparse_dir = output_dir / str(rec_id) # The warped input
                model_perfect_dir = undistorted_dir / str(rec_id) # The flawless output
                model_perfect_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    # 3. Undistort the Images & Rewrite Camera Math
                    pycolmap.undistort_images(
                        output_path=model_perfect_dir,
                        input_path=model_sparse_dir,
                        image_path=image_dir
                    )
                    print(f"  [SUCCESS] Images un-warped. Ready for 3DGS training at: {model_perfect_dir}")
                    
                    # 4. Export the True, Flat PLY for SuperSplat
                    # We must read from the NEW 'sparse' folder the undistorter just created
                    unwarped_sparse_dir = model_perfect_dir / "sparse"
                    unwarped_model = pycolmap.Reconstruction(unwarped_sparse_dir)
                    
                    unwarped_ply_path = model_perfect_dir / f"unwarped_model_{rec_id}.ply"
                    unwarped_model.export_PLY(str(unwarped_ply_path))
                    print(f"  [SUCCESS] Exported perfect geometry to: {unwarped_ply_path}")
                    
                except AttributeError:
                    print(f"\n[ERROR] PyCOLMAP is missing the 'undistort_images' binding. You must use the subprocess method instead.")
                    break  # Stop the loop, the API binding doesn't exist
                except Exception as e:
                    print(f"\n[ERROR] Undistorter failed on Model {rec_id}: {e}")

        else:
            print("\n[FAILED] Failed to create any sparse model. Check your baseline error margins or image overlap.")

    except Exception as e:
        print(f"\n[ERROR] PyCOLMAP Incremental Mapping Exception: {e}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--output_dir", default="sparse/0")
    parser.add_argument("--undistorted_dir", default="undistorted")

    return parser.parse_args()

if __name__ == "__main__":
    main()

# W20260330 18:36:50.839634 0x20980f100 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.