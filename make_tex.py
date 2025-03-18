import os
import re
import sys
import math
import shutil
from collections import defaultdict

def make_tex(directory_path, output_filename="presentation.tex"):
    """
    Generate a LaTeX beamer presentation with animated frames from a directory of images.
    Each environment (unique iter, env, run-id combination) gets its own GIF.
    
    Args:
        directory_path (str): Path to the directory containing the annotated images
        output_filename (str): Name of the output LaTeX file
    """
    print(f"Generating LaTeX file from images in {directory_path}")
    # Put the output file in the parent directory of the images
    if directory_path.endswith("/"):
        directory_path = directory_path[:-1]
    output_path = os.path.join(os.path.dirname(directory_path), output_filename)
    
    # Get all PNG files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.endswith(".png")]
    
    if len(image_files) == 0:
        print(f"No images found in {directory_path}")
        return
    
    # Single regex to extract all components at once
    file_pattern = re.compile(r'iter(\d+).*?env(\d+).*?step(\d+).*?run-id(\d+)')
    
    # Group files by environment
    env_groups = defaultdict(list)
    
    for filename in image_files:
        match = file_pattern.search(filename)
        
        # Skip files that don't match our pattern
        if not match:
            continue
        
        iter_num = match.group(1)
        env_num = match.group(2)
        step_num = int(match.group(3))
        run_id = match.group(4)
        
        # Create a key for this environment
        env_key = (iter_num, env_num, run_id)
        
        # Add the filename and step number to the appropriate environment group
        env_groups[env_key].append((filename, step_num))
    
    if not env_groups:
        print(f"No valid environment files found in {directory_path}")
        return
    
    # Sort the files in each environment group by step number
    for env_key in env_groups:
        env_groups[env_key].sort(key=lambda x: x[1])
    
    # Get the directory name for use in the LaTeX file
    dir_name = os.path.basename(os.path.normpath(directory_path))
    
    # Create the LaTeX preamble
    preamble = r"""\documentclass[pdf]{beamer}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{animate}
\usepackage[absolute,overlay]{textpos}
\mode<presentation>{}
\setlength{\fboxrule}{2pt}
%gets rid of bottom navigation symbols
\beamertemplatenavigationsymbolsempty
\setbeamertemplate{page number in head/foot}{}
%gets rid of footer
%will override 'frame number' instruction above
%comment out to revert to previous/default definitions
\newcommand\fps{14}

\begin{document}
"""
    
    # Create a temporary directory for symbolic links
    temp_dir = os.path.join(directory_path, "temp_links")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create the document body with frames
    body = ""
    for env_idx, (env_key, file_list) in enumerate(env_groups.items()):
        iter_num, env_num, run_id = env_key
        
        # Create a subdirectory for this environment
        env_subdir = f"iter{iter_num}_env{env_num}_run{run_id}"
        env_dir_path = os.path.join(temp_dir, env_subdir)
        os.makedirs(env_dir_path, exist_ok=True)
        
        # Create symbolic links with sequential numbers
        for i, (filename, _) in enumerate(file_list):
            source_path = os.path.join(directory_path, filename)
            link_name = f"{i}.png"
            link_path = os.path.join(env_dir_path, link_name)
            
            os.symlink(source_path, link_path)
        
        # Create a frame for this environment
        frame_title = f"Environment {env_idx}"
        max_frames = 500
        last_frame = min(len(file_list), max_frames) - 1 # -1 because animategraphics is inclusive
        
        frame = f"""
\\begin{{frame}}
\\frametitle{{{frame_title}}}
\\begin{{center}}
\\animategraphics[loop,controls,autoplay,height=2.1 in]{{\\fps}}{{{env_dir_path}/}}{{0}}{{{last_frame}}}
\\end{{center}}
\\end{{frame}}
"""
        
        body += frame
    
    # Add document closing
    closing = r"\end{document}"
    
    # Combine all parts
    full_document = preamble + body + closing
    
    # Write to output file
    with open(output_path, "w") as f:
        f.write(full_document)
    
    print(f"Generated {output_path} with {len(env_groups)} environments. Left temp links in {temp_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_tex.py <directory_path>")
        sys.exit(1)
    directory_path = sys.argv[1]
    make_tex(directory_path)
