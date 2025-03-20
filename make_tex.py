import os
import re
import sys
import math
import shutil
from collections import defaultdict
import subprocess

def make_tex(image_dir_path, output_filename="presentation.tex"):
    """
    Generate a LaTeX beamer presentation with animated frames from a directory of images.
    Each environment (unique iter, env, run-id combination) gets its own GIF.
    """
    print(f"Generating LaTeX file from images in {image_dir_path}")
    # Put the output file in the parent directory of the images
    if image_dir_path.endswith("/"):
        image_dir_path = image_dir_path[:-1]
    
    # Get all PNG files in the directory
    image_files = [f for f in os.listdir(image_dir_path) if f.endswith(".png")]
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir_path}")
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
        print(f"No valid environment files found in {image_dir_path}")
        return
    
    # Sort the files in each environment group by step number
    for env_key in env_groups:
        env_groups[env_key].sort(key=lambda x: x[1])
    
    # Get the directory name for use in the LaTeX file
    dir_name = os.path.basename(os.path.normpath(image_dir_path))
    
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
\newcommand\fps{12}

\begin{document}
"""
    
    # Create a temporary directory for temporary copies (because animategraphics wants files named something like 0.png, 1.png, etc.)
    temp_dir = os.path.join(image_dir_path, "temp_copies")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create the document body with frames in sorted order
    body = ""
    for env_idx, (env_key, file_list) in enumerate(sorted(env_groups.items())):
        iter_num, env_num, run_id = env_key
        
        # Create a subdirectory for this environment
        env_subdir = f"iter{iter_num}_env{env_num}_run{run_id}"
        env_dir_path = os.path.join(temp_dir, env_subdir)
        os.makedirs(env_dir_path, exist_ok=True)
        
        for i, (filename, _) in enumerate(file_list):
            source_path = os.path.join(image_dir_path, filename)
            copy_path = os.path.join(env_dir_path, f"{i}.png")
            shutil.copy2(source_path, copy_path)
        
        # Create a frame for this environment
        frame_title = f"Environment {env_idx + 1}"
        max_frames = 300
        last_frame = min(len(file_list), max_frames) - 1 # -1 because animategraphics is inclusive
        frame_prefix = os.path.join(os.path.basename(image_dir_path), 'temp_copies', env_subdir)
        frame = f"""
\\begin{{frame}}
\\frametitle{{{frame_title}}}
\\begin{{center}}
\\animategraphics[loop,controls,autoplay,height=2.1 in]{{\\fps}}{{{frame_prefix}/}}{{0}}{{{last_frame}}}
\\end{{center}}
\\end{{frame}}
"""
        
        body += frame
    
    # Add document closing
    closing = r"\end{document}"
    
    # Combine all parts
    full_document = preamble + body + closing
    
    # Write to output file
    output_dir = os.path.dirname(image_dir_path)
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        f.write(full_document)
    
    print(f"Generated {output_path} with {len(env_groups)} environments. Left temporary copies in {temp_dir}")
    # Compile the tex file. For some reason we need to do it twice to make the gifs work
    print("First tex compilation...")
    subprocess.run(f"cd {output_dir}; pdflatex {output_filename}", shell=True, stdout=subprocess.DEVNULL)
    print("Second tex compilation...")
    subprocess.run(f"cd {output_dir}; pdflatex {output_filename}", shell=True, stdout=subprocess.DEVNULL)
    # Remove temporary copies
    print(f"Removing temporary copies in {temp_dir}")
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python make_tex.py <image_dir_path> <output_filename>")
        sys.exit(1)
    image_dir_path = sys.argv[1]
    output_filename = sys.argv[2]
    make_tex(image_dir_path, output_filename)
