import os
import re
import sys
import time
import concurrent.futures
from tqdm import tqdm

def sort_key(filename):
    if filename.startswith("temp_"):
        stem = filename.split('.')[0]
        return int(stem.split('_')[1])
    else:
        return int(filename.split('.')[0])
    
def rename_file(args):
    """Rename a single file with error handling"""
    src_path, dst_path = args
    try:
        os.rename(src_path, dst_path)
        return True
    except Exception as e:
        print(f"Error renaming {src_path} to {dst_path}: {str(e)}")
        return False

def rename_files_sequentially(directory='.', num_workers=32):
    print(f"Scanning directory: {directory}")
    start_time = time.time()
    
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    total_files = len(files)
    scan_time = time.time() - start_time
    print(f"Found {total_files} matching files (scan took {scan_time:.2f} seconds).")
    
    if not files:
        print("No files to rename.")
        return
    
    # Sort files by their numeric value
    print("Sorting files...")
    files.sort(key = sort_key)
    files = [(sort_key(f), f) for f in files]
    
    # Determine which files need to be renamed
    rename_map = {}
    for new_number, (old_number, filename) in enumerate(files):
        if old_number != new_number:
            rename_map[filename] = f"{new_number}.png"
    
    rename_count = len(rename_map)
    if rename_count == 0:
        print("All files are already sequentially numbered. No renaming needed.")
        return
    
    print(f"{rename_count} out of {total_files} files need to be renamed using {num_workers} parallel workers.")
    
    # Use a safe two-phase renaming strategy with temporary filenames
    temp_prefix = "_temp_rename_"
    
    # First phase: rename to temporary names (in parallel)
    print("Phase 1: Renaming files to temporary names...")
    phase1_tasks = []
    
    for old_filename, new_filename in rename_map.items():
        old_path = os.path.join(directory, old_filename)
        temp_path = os.path.join(directory, f"{temp_prefix}{old_filename}")
        phase1_tasks.append((old_path, temp_path))
    
    # Execute phase 1 with progress bar
    successful_phase1 = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(rename_file, phase1_tasks),
            total=len(phase1_tasks),
            desc="Phase 1",
            unit="files"
        ))
        successful_phase1 = sum(results)
    
    # Second phase: rename from temporary to final names (in parallel)
    print("Phase 2: Renaming files to final names...")
    phase2_tasks = []
    
    for old_filename, new_filename in rename_map.items():
        temp_path = os.path.join(directory, f"{temp_prefix}{old_filename}")
        new_path = os.path.join(directory, new_filename)
        # Only add to phase 2 if temp file exists
        if os.path.exists(temp_path):
            phase2_tasks.append((temp_path, new_path))
    
    # Execute phase 2 with progress bar
    successful_phase2 = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(rename_file, phase2_tasks),
            total=len(phase2_tasks),
            desc="Phase 2",
            unit="files"
        ))
        successful_phase2 = sum(results)
    
    total_time = time.time() - start_time
    print(f"Operation completed in {total_time:.2f} seconds.")
    print(f"Successfully renamed {successful_phase2} files.")
    if successful_phase2 != rename_count:
        print(f"Warning: {rename_count - successful_phase2} files couldn't be renamed properly.")

if __name__ == "__main__":
    # Get directory from command line args or use current directory
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    # Get number of workers from command line or use default
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    
    rename_files_sequentially(directory, num_workers)
