import os
import sys
from datetime import datetime

def find_files_with_keyword(directory, keyword):
    
    # Get all files in the directory
    all_files = []
    try:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            # Check if it's a file and contains the keyword
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    try:
                        if keyword in f.read():
                            # Get last modified time
                            mod_time = os.path.getmtime(file_path)
                            all_files.append((file_path, mod_time))
                    except UnicodeDecodeError:
                        print(f"Could not read file: {file_path}")
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return []
    except PermissionError:
        print(f"Permission denied: {directory}")
        return []
    
    # Sort by modification time (oldest first)
    all_files.sort(key=lambda x: x[1])
    return all_files

def main():
    # Check arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py <directory> <keyword>")
        return
    
    directory = sys.argv[1]
    keyword = sys.argv[2]
    
    # Find files with keyword
    matching_files = find_files_with_keyword(directory, keyword)
    
    if not matching_files:
        print(f"No files containing '{keyword}' found in {directory}")
        return
    
    # Print results
    print(f"Files containing '{keyword}' in {directory}, ordered by modification date:")
    for file_path, mod_time in matching_files:
        mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{mod_time_str} - {file_path}")

if __name__ == "__main__":
    main()
