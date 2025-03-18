import os
import glob
import argparse
import re

def process_images(directory, n):
    # Get all PNG files in the directory
    png_pattern = os.path.join(directory, '*.png')
    png_files = sorted(glob.glob(png_pattern), key=lambda x: int(re.search(r'(\d+)\.png$', x).group(1)))
    
    if not png_files:
        print(f"No PNG files found in {directory}")
        return
    
    if n > len(png_files):
        print(f"Warning: Requested to delete {n} files, but only {len(png_files)} PNG files exist")
        n = len(png_files)
    
    # Delete the first n PNG files
    for i in range(n):
        print(f"Deleting {png_files[i]}")
        os.remove(png_files[i])
    
    # Rename the remaining files
    remaining_files = png_files[n:]
    
    # Create a mapping of original files to new names
    rename_map = {}
    for new_idx, old_path in enumerate(remaining_files):
        new_filename = f"{new_idx}.png"
        new_path = os.path.join(directory, new_filename)
        rename_map[old_path] = new_path
    
    # Apply renames (using temporary names first to avoid conflicts)
    for old_path, new_path in rename_map.items():
        temp_path = old_path + ".tmp"
        os.rename(old_path, temp_path)
    
    for old_path, new_path in rename_map.items():
        temp_path = old_path + ".tmp"
        os.rename(temp_path, new_path)
    
    print(f"Deleted {n} files and renamed {len(remaining_files)} files to start from 0.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete first n PNG files and rename the rest starting from 0.png")
    parser.add_argument("directory", help="Directory containing PNG files")
    parser.add_argument("n", type=int, help="Number of PNG files to delete from the beginning")
    
    args = parser.parse_args()
    
    process_images(args.directory, args.n)
