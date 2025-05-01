# preprocess_data.py
import os
import shutil
import argparse
import sys
from pathlib import Path

# Add project root and relevant submodules to Python path
# Adjust these paths if the script is not run from the 'notebooks' directory
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "submodules/gaussian-splatting"))

# Try importing necessary preprocessing functions
try:
    from source.utils_preprocess import (
        read_video_frames,
        preprocess_frames, # Assuming this might exist for resizing/cropping if needed
        select_optimal_frames,
        save_frames_to_scene_dir,
        run_colmap_on_scene
    )
    print("Successfully imported preprocessing utilities.")
    preprocessing_available = True
except ImportError as e:
    print(f"Warning: Could not import preprocessing utilities: {e}")
    print("Video preprocessing function will not be available.")
    print("Ensure you are running from the correct directory and submodules are installed.")
    preprocessing_available = False

def preprocess_video_data(video_path, output_dir, num_frames):
    """
    Processes a video file: extracts frames, selects optimal ones, saves them,
    and runs COLMAP.
    """
    if not preprocessing_available:
        print("Error: Preprocessing functions are not available. Cannot process video.")
        return

    print(f"Starting video preprocessing for: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of frames to extract: {num_frames}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Read Video Frames
        print("Step 1: Reading video frames...")
        frames_data = read_video_frames(video_path)
        print(f"Read {len(frames_data)} frames.")

        # 2. Preprocess Frames (Optional - Placeholder if needed)
        # frames_data = preprocess_frames(frames_data, ...) # e.g., resizing

        # 3. Select Optimal Frames
        print("Step 2: Selecting optimal frames...")
        selected_frames_indices = select_optimal_frames(frames_data, num_frames)
        selected_frames = [frames_data[i] for i in selected_frames_indices]
        print(f"Selected {len(selected_frames)} frames.")

        # 4. Save Frames to Scene Directory
        print(f"Step 3: Saving selected frames to {output_dir}...")
        # save_frames_to_scene_dir expects the *parent* of the 'images' folder
        save_frames_to_scene_dir(selected_frames, output_dir)
        print("Frames saved.")

        # 5. Run COLMAP
        print("Step 4: Running COLMAP...")
        # run_colmap_on_scene expects the scene directory (parent of 'images')
        run_colmap_on_scene(output_dir)
        print("COLMAP processing finished.")

        print(f"Video preprocessing complete. Output saved to: {output_dir}")

    except Exception as e:
        print(f"An error occurred during video preprocessing: {e}")
        # Optional: Clean up partial results
        # shutil.rmtree(output_dir, ignore_errors=True)

def preprocess_image_data(image_dir, output_dir):
    """
    Processes an image directory by running COLMAP.
    Assumes the input directory contains the images.
    Copies the structure to the output directory and runs COLMAP there.
    """
    print(f"Starting image preprocessing for directory: {image_dir}")
    print(f"Output directory: {output_dir}")

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    if not image_dir.is_dir():
        print(f"Error: Input image directory not found: {image_dir}")
        return

    # Create output structure (copy images)
    output_image_subdir = output_dir / "images"
    try:
        # Copy images to output_dir/images, mimicking standard structure
        if output_dir.exists():
             print(f"Warning: Output directory {output_dir} already exists. Overwriting images subdirectory.")
             shutil.rmtree(output_image_subdir, ignore_errors=True)
        shutil.copytree(image_dir, output_image_subdir)
        print(f"Copied images from {image_dir} to {output_image_subdir}")
    except Exception as e:
        print(f"Error copying images: {e}")
        return

    # Check if COLMAP already ran in source (heuristic)
    if (image_dir.parent / "sparse" / "0").exists() and (image_dir.parent / "sparse" / "0").is_dir():
         print("Detected existing COLMAP results in source parent directory. Copying...")
         source_sparse_dir = image_dir.parent / "sparse"
         dest_sparse_dir = output_dir / "sparse"
         try:
             if dest_sparse_dir.exists():
                  shutil.rmtree(dest_sparse_dir)
             shutil.copytree(source_sparse_dir, dest_sparse_dir)
             print(f"Copied sparse reconstruction from {source_sparse_dir} to {dest_sparse_dir}")
             print("Image preprocessing complete (copied existing COLMAP).")
             return
         except Exception as e:
             print(f"Warning: Failed to copy existing COLMAP results: {e}. Will attempt to run COLMAP.")


    # Run COLMAP on the *output* directory
    print("Running COLMAP...")
    try:
        # run_colmap_on_scene expects the scene directory (parent of 'images')
        run_colmap_on_scene(output_dir)
        print("COLMAP processing finished.")
        print(f"Image preprocessing complete. Output saved to: {output_dir}")
    except Exception as e:
        print(f"An error occurred during COLMAP execution: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess video or image data for EDGS.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input video file or image directory.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save processed data. Default is input path.")
    parser.add_argument("--type", type=str, choices=['video', 'image'], required=True, help="Type of input data.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to extract for video data.")
    
    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = args.input

    if args.type == 'video':
        if not Path(args.input).is_file():
            print(f"Error: Video file not found: {args.input}")
        else:
            preprocess_video_data(args.input, args.output_dir, args.num_frames)
    elif args.type == 'image':
        if not Path(args.input).is_dir():
            print(f"Error: Image directory not found: {args.input}")
        else:
            preprocess_image_data(args.input, args.output_dir)

    print("Preprocessing script finished.")