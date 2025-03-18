#!/usr/bin/env python3
import os
import argparse
import concurrent.futures
import shutil
import re
from pathlib import Path
from PIL import Image
import time

def validate_image(img_path):
    """Validate a single image file."""
    try:
        with Image.open(img_path) as img:
            # Force PIL to fully load the image to check for corruption
            img.verify()
            # Additional validation: try loading the image again and accessing its data
            with Image.open(img_path) as img2:
                img2.load()
        return (img_path, True, None)
    except Exception as e:
        return (img_path, False, str(e))

def extract_number(filepath):
    """Extract number from filename for sorting."""
    base_name = Path(filepath).stem  # Get filename without extension
    # Extract all digits from the filename
    return int(digits)

def main():
    parser = argparse.ArgumentParser(description='Scan a dataset for corrupted images')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--output', type=str, default='corrupted_images.txt',
                        help='File to save the list of corrupted images')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker processes')
    parser.add_argument('--delete', action='store_true',
                        help='Delete corrupted images after scanning')
    parser.add_argument('--renumber', action='store_true',
                        help='Renumber all images sequentially after deletion')
    parser.add_argument('--force', action='store_true',
                        help='Skip confirmation prompt for deletion and renumbering')
    args = parser.parse_args()
    
    # Get all image files
    data_dir = Path(args.data_dir)
    image_files = list(data_dir.glob('**/*.png')) + list(data_dir.glob('**/*.jpg'))
    
    print(f"Found {len(image_files)} image files in {data_dir}")
    print(f"Validating images using {args.workers} workers...")
    
    start_time = time.time()
    corrupted_images = []
    processed = 0
    
    # Process files in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(validate_image, img_path) for img_path in image_files]
        
        for future in concurrent.futures.as_completed(futures):
            img_path, valid, error = future.result()
            processed += 1
            
            # Progress update every 1000 images
            if processed % 10000 == 0:
                elapsed = time.time() - start_time
                images_per_second = processed / elapsed
                print(f"Processed {processed}/{len(image_files)} images ({images_per_second:.2f} images/sec)")
            
            if not valid:
                print(f"Corrupted image: {img_path}, Error: {error}")
                corrupted_images.append((img_path, error))
    
    # Save results
    with open(args.output, 'w') as f:
        f.write(f"# Found {len(corrupted_images)} corrupted images out of {len(image_files)}\n")
        for img_path, error in corrupted_images:
            f.write(f"{str(img_path)}\t{error}\n")
    
    print(f"\nScan completed in {time.time() - start_time:.2f} seconds")
    print(f"Found {len(corrupted_images)} corrupted images out of {len(image_files)} ({len(corrupted_images)/len(image_files)*100:.2f}%)")
    print(f"Results saved to {args.output}")
    
    # Track if we need to renumber after this operation
    need_renumber = args.renumber
    deleted_files = False
    
    # Delete corrupted images if requested
    if args.delete and corrupted_images:
        if not args.force:
            print("\n" + "!"*80)
            print(f"WARNING: You are about to delete {len(corrupted_images)} corrupted images.")
            print("This operation cannot be undone!")
            print("!"*80 + "\n")
            
            confirm = input("Type 'yes' to confirm deletion: ")
            if confirm.lower() != 'yes':
                print("Deletion cancelled.")
                need_renumber = False  # Skip renumbering if deletion was cancelled
            else:
                deleted_files = True
        else:
            deleted_files = True
            
        if deleted_files:
            print(f"Deleting {len(corrupted_images)} corrupted images...")
            deleted_count = 0
            failed_count = 0
            failed_files = []
            
            for img_path, _ in corrupted_images:
                try:
                    Path(img_path).unlink()
                    deleted_count += 1
                    
                    # Progress update every 100 deletions
                    if deleted_count % 100 == 0:
                        print(f"Deleted {deleted_count}/{len(corrupted_images)} files")
                except Exception as e:
                    failed_count += 1
                    failed_files.append((img_path, str(e)))
                    print(f"Failed to delete {img_path}: {e}")
            
            print(f"\nDeletion complete: {deleted_count} files deleted, {failed_count} deletion failures")
            
            # Log deletion failures if any
            if failed_count > 0:
                failure_log = "deletion_failures.txt"
                with open(failure_log, 'w') as f:
                    f.write(f"# Failed to delete {failed_count} corrupted images\n")
                    for img_path, error in failed_files:
                        f.write(f"{img_path}\t{error}\n")
                print(f"Deletion failures logged to {failure_log}")
    
    # Renumber files sequentially if requested and needed
    if need_renumber and (deleted_files or args.force):
        if not args.force:
            print("\n" + "!"*80)
            print("WARNING: You are about to renumber all images sequentially.")
            print("This will rename all files in the dataset!")
            print("!"*80 + "\n")
            
            confirm = input("Type 'yes' to confirm renumbering: ")
            if confirm.lower() != 'yes':
                print("Renumbering cancelled.")
                return
        
        print("\nRenumbering all images sequentially...")
        
        # Get all remaining image files
        image_files = list(data_dir.glob('**/*.png')) + list(data_dir.glob('**/*.jpg'))
        image_files.sort(key=extract_number)  # Sort by existing number
        
        # Create a temporary directory for renaming
        temp_dir = data_dir / "temp_rename"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Copy files with new sequential names to temp directory
            total_files = len(image_files)
            digits = len(str(total_files))  # Number of digits needed for padding
            
            print(f"Moving {total_files} files to temporary directory...")
            
            for idx, img_path in enumerate(image_files):
                ext = img_path.suffix
                new_name = f"{idx+1:0{digits}d}{ext}"  # New sequential filename with padding
                target_path = temp_dir / new_name
                
                # Copy file to temp directory with new name
                shutil.copy2(img_path, target_path)
                
                # Progress update
                if (idx+1) % 1000 == 0 or idx+1 == total_files:
                    print(f"Copied {idx+1}/{total_files} files")
            
            # Delete original files
            print(f"Removing original files...")
            for img_path in image_files:
                img_path.unlink()
            
            # Move renamed files back to original directory
            print(f"Moving renamed files back to original directory...")
            for idx, img_path in enumerate(sorted(temp_dir.glob('*'))):
                target_path = data_dir / img_path.name
                shutil.move(img_path, target_path)
                
                # Progress update
                if (idx+1) % 1000 == 0 or idx+1 == total_files:
                    print(f"Moved {idx+1}/{total_files} files")
            
            print(f"\nRenumbering complete! {total_files} files have been sequentially renumbered.")
        
        except Exception as e:
            print(f"Error during renumbering: {e}")
            print("Some files may be in the temporary directory. Manual cleanup may be required.")
        finally:
            # Clean up temporary directory
            try:
                if temp_dir.exists():
                    for f in temp_dir.glob('*'):
                        f.unlink()
                    temp_dir.rmdir()
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")

if __name__ == "__main__":
    main()
