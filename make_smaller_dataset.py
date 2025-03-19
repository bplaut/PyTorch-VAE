import os
import re
import random
import shutil
import argparse
from collections import defaultdict

def copy_random_environments(input_dir, output_dir, num_envs, seed=None):
    """
    Copy files for n randomly selected environments from input_dir to output_dir.
    An environment is defined by a unique combination of iter, env, and run-id.
    """
    if seed is not None:
        random.seed(seed)    
    os.makedirs(output_dir, exist_ok=True)
    png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    env_pattern = re.compile(r'iter(\d+).*?env(\d+).*?run-id(\d+)')
    
    # Group files by environment (iter, env, run-id)
    env_groups = defaultdict(list)
    print("Grouping files by environment...")
    for filename in png_files:
        # Use the regex to extract environment components
        match = env_pattern.search(filename)
        
        # Skip files that don't match our pattern (including simple filenames)
        if not match:
            continue
        
        iter_num = match.group(1)
        env_num = match.group(2)
        run_id = match.group(3)
        
        # Create a key for this environment
        env_key = (iter_num, env_num, run_id)
        
        # Add the filename to the appropriate environment group
        env_groups[env_key].append(filename)
    
    # Select n random environments
    if num_envs >= len(env_groups):
        raise ValueError(f"Requested {num_envs} environments, but only {len(env_groups)} available")
    else:
        print(f"Found {len(env_groups)} environments (aka levels)")
        selected_envs = random.sample(list(env_groups.keys()), num_envs)
    
    # Copy all files for the selected environments
    copied_files = 0
    for env_key in selected_envs:
        for filename in env_groups[env_key]:
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            shutil.copy2(src_path, dst_path)
            copied_files += 1
    
    return len(selected_envs), copied_files

def main():
    parser = argparse.ArgumentParser(description='Copy files from random environments')
    parser.add_argument('input_dir', help='Directory containing the input PNG files')
    parser.add_argument('output_dir', help='Directory where selected files will be copied')
    parser.add_argument('num_envs', type=int, help='Number of environments to randomly select')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    num_envs, num_files = copy_random_environments(
        args.input_dir, args.output_dir, args.num_envs, args.seed
    )
    
    print(f"Selected {num_envs} environments")
    print(f"Copied {num_files} files to {args.output_dir}")

if __name__ == '__main__':
    main()
