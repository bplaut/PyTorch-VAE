import os
import argparse
from collections import defaultdict

def find_all_version_epochs(root_dir):
    # Dictionary to track models, versions and file counts
    model_data = defaultdict(lambda: defaultdict(int))
    
    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not 'train' in dirpath:
            continue
        if os.path.basename(dirpath) == "reconstructions" or os.path.basename(dirpath) == "highest_and_lowest_loss_imgs":
            # Extract model name and version from path
            path_parts = dirpath.split(os.sep)
            if len(path_parts) >= 3:  # Ensure we have enough parts for model/version/reconstructions
                model_idx = path_parts.index(root_dir.rstrip(os.sep).split(os.sep)[-1]) + 1 if root_dir in dirpath else 0
                if model_idx < len(path_parts) - 2:
                    model_name = path_parts[model_idx]
                    version_name = path_parts[model_idx + 1]
                    # Count files in this directory
                    if os.path.basename(dirpath) == "reconstructions":
                        file_count = len(filenames)
                    else:
                        file_count = len([f for f in filenames if "highest" in f]) # This directory has two files per epoch: highest and lowest
                    if file_count > model_data.get(model_name, {}).get(version_name, 0):
                        model_data[model_name][version_name] = file_count
    
    return model_data

def main():
    parser = argparse.ArgumentParser(description='Find epochs completed for all versions of each model')
    parser.add_argument('directory', help='Root directory to search')
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return
    
    model_data = find_all_version_epochs(args.directory)
    
    print("Results:")
    for model, versions in sorted(model_data.items()):
        print(f"{model}:")
        for version, count in sorted(versions.items()):
            print(f"  {version}: {count} epochs")

if __name__ == "__main__":
    main()
