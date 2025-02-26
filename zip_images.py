import os
import zipfile

def zip_directory(source_dir, output_filename, exclude_dirs):
    """
    Create a zip file from a directory, excluding specified subdirectories.
    
    Args:
        source_dir (str): The directory to zip
        output_filename (str): The name of the output zip file
        exclude_dirs (list): List of directory names to exclude
    """
    if exclude_dirs is None:
        exclude_dirs = []
    
    # Convert to absolute paths for reliable comparisons
    source_dir = os.path.abspath(source_dir)
    
    # Create the zip file
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the directory tree
        for root, dirs, files in os.walk(source_dir):
            # Remove excluded directories from the dirs list to avoid walking into them
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            # Add all files to the zip
            for file in files:
                file_path = os.path.join(root, file)
                # Make the path in the zip file relative to the source directory
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Zip a directory while excluding specified subdirectories')
    parser.add_argument('source_dir', help='Source directory to zip')
    parser.add_argument('output_zip', help='Output zip file name')
    parser.add_argument('--exclude', nargs='+', default=['checkpoints'], 
                        help='Directory names to exclude (default: checkpoints)')
    
    args = parser.parse_args()
    
    print(f"Zipping {args.source_dir} to {args.output_zip}, excluding {args.exclude}")
    zip_directory(args.source_dir, args.output_zip, args.exclude)
    print("Done")
