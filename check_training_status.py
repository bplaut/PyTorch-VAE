import os
import argparse
from collections import defaultdict

def find_versions_with_most_reconstructions(root_dir):
    # Dictionary to track models, versions and file counts
    model_data = defaultdict(lambda: defaultdict(int))
    
    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not 'train' in dirpath:
            continue
        if os.path.basename(dirpath) == "reconstructions":
            # Extract model name and version from path
            path_parts = dirpath.split(os.sep)
            if len(path_parts) >= 3:  # Ensure we have enough parts for model/version/reconstructions
                model_idx = path_parts.index(root_dir.rstrip(os.sep).split(os.sep)[-1]) + 1 if root_dir in dirpath else 0
                if model_idx < len(path_parts) - 2:
                    model_name = path_parts[model_idx]
                    version_name = path_parts[model_idx + 1]
                    # Count files in this reconstruction directory
                    file_count = len(filenames)
                    model_data[model_name][version_name] = file_count
    
    # Find the version with most files for each model
    results = {}
    for model, versions in model_data.items():
        if versions:  # Only process if we have data
            max_version = max(versions.items(), key=lambda x: x[1])
            results[model] = max_version
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Find which version of each model has the most reconstruction files')
    parser.add_argument('directory', help='Root directory to search')
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return
    
    results = find_versions_with_most_reconstructions(args.directory)
    
    if results:
        print("Results:")
        for model, (version, count) in sorted(results.items()):
            print(f"{model}: {version} completed {count} epochs")
    else:
        print(f"No reconstruction directories found in {args.directory}")

if __name__ == "__main__":
    main()
