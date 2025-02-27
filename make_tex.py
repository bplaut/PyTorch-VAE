import os
import math
import sys

def make_tex(directory_path, output_filename="presentation.tex"):
    """
    Generate a LaTeX beamer presentation with animated frames from a directory of images.
    
    Args:
        directory_path (str): Path to the directory containing the annotated images
        output_filename (str): Name of the output LaTeX file
    """
    print(f"Generating LaTeX file from images in {directory_path}")
    # Put the output file in the parent directory of the images
    if directory_path.endswith("/"):
        directory_path = directory_path[:-1]
    output_path = os.path.join(os.path.dirname(directory_path), output_filename)
    
    # Count the number of image files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.endswith(".png")]
    num_images = len(image_files)
    
    if num_images == 0:
        print(f"No images found in {directory_path}")
        return
    
    # Find the highest image number to ensure we don't exceed it
    max_image_num = max([int(f.split('.')[0]) for f in image_files])
    
    # Calculate the number of frames needed (500 images per frame)
    num_frames = math.ceil(max_image_num / 500)
    
    # Get the directory name for use in the LaTeX file
    dir_name = os.path.basename(os.path.normpath(directory_path))
    
    # Create the LaTeX preamble
    preamble = r"""\documentclass[pdf]{beamer}
\usepackage[utf8]{inputenc}
\usepackage{gensymb}
\usepackage{amsmath,amsfonts}
\usepackage{graphicx}
\usepackage{pgfplots}
\usepackage{tikz}
\usepackage{mathtools}
\usepackage{amsthm}
\usepackage{nccmath}
\usepackage[percent]{overpic}
\usepackage{rotating}
\usepackage{array, booktabs}
\usepackage{centernot}
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
\newcommand\fps{8}

\begin{document}
"""
    
    # Create the document body with frames
    body = ""
    for i in range(num_frames):
        start_idx = i * 500
        end_idx = min((i + 1) * 500, max_image_num)
        
        # Skip this frame if start_idx exceeds our max image number
        if start_idx > max_image_num:
            break
            
        frame = f"""
\\begin{{frame}}
\\begin{{center}}
\\animategraphics[loop,controls,autoplay,height=2.1 in]{{\\fps}}{{{dir_name}/}}{{{start_idx}}}{{{end_idx}}}
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
    
    print(f"LaTeX file generated: {output_path}")
    print(f"Found {max_image_num + 1} images, created {num_frames} animation frames")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_tex.py <directory_path>")
        sys.exit(1)
    directory_path = sys.argv[1]
    make_tex(directory_path)
