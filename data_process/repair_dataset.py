import json
import os
from pathlib import Path

def repair_dataset(dataset_dir):
    """
    Repairs the dataset by:
    1. Removing entries where images are missing (based on <image> tag vs empty images list, or file existence).
    2. (If path-based) Removing orphaned image files that are not referenced in the JSON.
    """
    splits = ['train', 'test', 'valid']
    dataset_path = Path(dataset_dir)
    
    print(f"Repairing dataset at {dataset_path}...")
    
    for split in splits:
        split_dir = dataset_path / split
        json_path = split_dir / "data.json"
        
        if not json_path.exists():
            print(f"Skipping {split}: data.json not found.")
            continue
            
        print(f"Processing {split}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        new_data = []
        removed_count = 0
        
        # Determine if the dataset uses file paths or base64 strings
        is_base64 = False
        for item in data:
            if item.get('images'):
                first_img = item['images'][0]
                # Simple heuristic: if length > 500 and doesn't look like a filename
                if len(first_img) > 500 or not (first_img.lower().endswith('.png') or first_img.lower().endswith('.jpg') or first_img.lower().endswith('.jpeg')):
                    is_base64 = True
                break
        
        if is_base64:
            print("  Detected Base64 encoded images.")
        else:
            print("  Detected file path references.")
            
        valid_images = set()
        
        for item in data:
            # 1. Extract prompt text to check for <image> tag
            prompt_text = ""
            if 'prompt' in item and isinstance(item['prompt'], list):
                for p in item['prompt']:
                    if p.get('role') == 'user':
                        prompt_text = p.get('content', "")
                        break
            elif 'conversations' in item: # Legacy/ShareGPT format
                for c in item['conversations']:
                    if c.get('from') == 'human':
                        prompt_text = c.get('value', "")
                        break
            
            has_image_tag = "<image>" in prompt_text
            images_list = item.get('images', [])
            
            # 2. Check for inconsistency: <image> tag present but no images
            if has_image_tag and not images_list:
                removed_count += 1
                continue
            
            # 3. If path-based, check if referenced files actually exist
            if not is_base64 and images_list:
                all_exist = True
                for img_path in images_list:
                    # Path in JSON is typically relative to split directory (e.g. images/xxx.png)
                    full_path = split_dir / img_path
                    if not full_path.exists():
                        # Try relative to dataset root just in case
                        full_path_alt = dataset_path / img_path
                        if full_path_alt.exists():
                            full_path = full_path_alt
                        else:
                            all_exist = False
                            break
                    
                    if all_exist:
                        valid_images.add(full_path.resolve())
                
                if not all_exist:
                    removed_count += 1
                    continue

            new_data.append(item)
            
        print(f"  Removed {removed_count} entries due to missing images.")
        
        if removed_count > 0:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)
            print(f"  Updated {json_path}")
        else:
            print("  No entries removed.")
            
        # 4. Remove orphaned images (Only for path-based)
        if not is_base64:
            images_dir = split_dir / "images"
            if images_dir.exists():
                removed_files = 0
                total_files = 0
                for root, _, files in os.walk(images_dir):
                    for file in files:
                        total_files += 1
                        file_path = Path(root) / file
                        if file_path.resolve() not in valid_images:
                            try:
                                os.remove(file_path)
                                removed_files += 1
                            except Exception as e:
                                print(f"  Error removing {file_path}: {e}")
                                
                # Clean up empty directories
                for root, dirs, _ in os.walk(images_dir, topdown=False):
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except OSError:
                            pass # Directory not empty
                            
                print(f"  Cleaned up images directory: Removed {removed_files} orphaned files out of {total_files}.")
        else:
            print("  Skipping orphaned file cleanup because images are embedded as Base64.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Repair dataset by removing entries with missing images and cleaning orphaned files.")
    parser.add_argument("--dataset_dir", type=str, default=r"E:\套瓷三剑客\math-pro\dataset-verl", help="Path to the dataset directory")
    args = parser.parse_args()
    
    repair_dataset(args.dataset_dir)
