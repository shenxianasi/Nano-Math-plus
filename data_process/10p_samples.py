import os
import json
import random
import shutil
from pathlib import Path

# Configuration
SOURCE_ROOT = Path(r"E:\套瓷三剑客\math-pro\dataset-SFT")
TARGET_ROOT = Path(r"E:\套瓷三剑客\math-pro\dataset-verl")
SAMPLE_RATE = 0.1
SPLITS = ["test", "valid", "train"]
RANDOM_SEED = 42

import sys

def process_split(split_name):
    """
    Process a single data split:
    1. Load data
    2. Sample 10%
    3. Save sampled data
    4. Copy associated images
    """
    source_split_dir = SOURCE_ROOT / split_name
    target_split_dir = TARGET_ROOT / split_name
    
    if not source_split_dir.exists():
        print(f"Skipping {split_name}: Source directory not found at {source_split_dir}")
        sys.stdout.flush()
        return

    json_path = source_split_dir / "data.json"
    if not json_path.exists():
        print(f"Skipping {split_name}: data.json not found at {json_path}")
        sys.stdout.flush()
        return
        
    print(f"Processing {split_name}...")
    sys.stdout.flush()
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        sys.stdout.flush()
        return
    
    total_count = len(data)
    sample_count = int(total_count * SAMPLE_RATE)
    
    if sample_count == 0 and total_count > 0:
        sample_count = 1  # Ensure at least 1 sample if data exists
    
    # Stratified sampling: 
    # Since we are processing each split (train/test/valid) independently and taking 10% from each,
    # we are maintaining the original distribution of splits.
    # Within each split, we use random sampling.
    sampled_data = random.sample(data, sample_count)
    
    print(f"  Sampled {sample_count} out of {total_count} items from {split_name}.")
    sys.stdout.flush()
    
    # Create target directory
    target_split_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sampled json FIRST to ensure we have it even if image copying fails/interrupts
    target_json_path = target_split_dir / "data.json"
    try:
        with open(target_json_path, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved sampled data to {target_json_path}")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error writing {target_json_path}: {e}")
        sys.stdout.flush()

    # Copy images and verify paths
    missing_images = 0
    copied_images = 0
    
    print("  Starting image copy...")
    sys.stdout.flush()
    
    for idx, item in enumerate(sampled_data):
        if idx % 100 == 0:
            print(f"    Processed {idx}/{len(sampled_data)} items for images...")
            sys.stdout.flush()
            
        if "images" in item and isinstance(item["images"], list):
            for img_rel_path in item["images"]:
                # img_rel_path e.g., "images/openmathreasoning_float_image/xxxxx.png"
                src_img = source_split_dir / img_rel_path
                dst_img = target_split_dir / img_rel_path
                
                if src_img.exists():
                    try:
                        dst_img.parent.mkdir(parents=True, exist_ok=True)
                        if not dst_img.exists():
                            shutil.copy2(src_img, dst_img)
                            copied_images += 1
                        else:
                            # Already exists (maybe referenced by multiple items)
                            pass
                    except Exception as e:
                        print(f"  Error copying image {src_img}: {e}")
                        sys.stdout.flush()
                else:
                    # Try to check if it's relative to root instead of split?
                    # But structure suggests it is relative to split folder.
                    missing_images += 1
                    # print(f"  Warning: Image not found {src_img}") 

    if missing_images > 0:
        print(f"  Warning: {missing_images} images were not found in source.")
    print(f"  Copied {copied_images} images.")
    sys.stdout.flush()


def main():
    random.seed(RANDOM_SEED)
    
    print(f"Starting 10% sampling from {SOURCE_ROOT} to {TARGET_ROOT}")
    
    if TARGET_ROOT.exists():
        print(f"Target directory {TARGET_ROOT} already exists. Merging/Overwriting...")
    else:
        TARGET_ROOT.mkdir(parents=True, exist_ok=True)
        
    for split in SPLITS:
        process_split(split)
        
    # Copy dataset_info.json if exists
    src_info = SOURCE_ROOT / "dataset_info.json"
    if src_info.exists():
        try:
            shutil.copy2(src_info, TARGET_ROOT / "dataset_info.json")
            print("Copied dataset_info.json")
        except Exception as e:
            print(f"Error copying dataset_info.json: {e}")
            
    print("Done.")

if __name__ == "__main__":
    main()
