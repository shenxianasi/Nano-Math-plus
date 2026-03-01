import os
import json
import base64
import re
import hashlib
from pathlib import Path
from tqdm import tqdm

def extract_answer(text):
    """
    Extract the answer from the \\boxed{...} format.
    Handles nested braces to some extent, but assumes the standard format.
    """
    if not text:
        return ""
    
    # Try to find the last \boxed{...}
    # A simple regex for non-nested braces
    # pattern = r"\\boxed\{([^}]*)\}"
    # matches = re.findall(pattern, text)
    # if matches:
    #     return matches[-1]
    
    # Better approach for nested braces:
    # Find position of \boxed{
    # Count braces to find the closing one.
    
    results = []
    idx = text.find("\\boxed{")
    while idx != -1:
        brace_count = 1
        start_content = idx + 7 # len("\\boxed{") is 7
        current = start_content
        content = ""
        while current < len(text) and brace_count > 0:
            if text[current] == '{':
                brace_count += 1
            elif text[current] == '}':
                brace_count -= 1
            
            if brace_count > 0:
                content += text[current]
            current += 1
            
        if brace_count == 0:
            results.append(content)
        
        # Search for next occurrence
        idx = text.find("\\boxed{", start_content)
        
    if results:
        return results[-1]
    
    return ""

def image_to_base64(image_path):
    """
    Read an image file and convert it to a base64 string.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def convert_dataset(dataset_dir):
    """
    Convert the dataset in dataset_dir to the Qwen2.5-VL RLHF format.
    Overwrites the data.json files in place.
    """
    dataset_path = Path(dataset_dir)
    splits = ['train', 'test', 'valid']
    
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
        
        for idx, item in tqdm(enumerate(data), total=len(data), desc=f"Converting {split}"):
            # Extract conversations
            conversations = item.get('conversations', [])
            if not conversations:
                continue
                
            # Assume first turn is user, second is gpt
            user_msg = next((c for c in conversations if c['from'] == 'human'), None)
            gpt_msg = next((c for c in conversations if c['from'] == 'gpt'), None)
            
            if not user_msg or not gpt_msg:
                continue
                
            prompt_content = user_msg['value']
            response_content = gpt_msg['value']
            
            # Extract ground truth
            ground_truth = extract_answer(response_content)
            
            # Handle images
            encoded_images = []
            if 'images' in item:
                for img_rel_path in item['images']:
                    # The image path in item is relative to the split directory?
                    # In the previous script, we copied images to dataset-verl/split/images/...
                    # So the path in json "images/..." is relative to split_dir.
                    img_full_path = split_dir / img_rel_path
                    
                    if img_full_path.exists():
                        b64_str = image_to_base64(img_full_path)
                        if b64_str:
                            encoded_images.append(b64_str)
                    else:
                        print(f"Warning: Image not found at {img_full_path}")
            
            # Construct new item
            new_item = {
                "data_source": item.get('metadata', {}).get('source', 'unknown'),
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_content
                    }
                ],
                "images": encoded_images,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": ground_truth,
                    "question": prompt_content
                }
            }
            
            new_data.append(new_item)
            
        # Overwrite the file
        print(f"Saving converted data to {json_path}...")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
            
    print("Conversion complete.")

def build_image_index(images_dir):
    """
    Builds a dictionary mapping SHA256 hash of image content to its relative path.
    """
    index = {}
    images_path = Path(images_dir)
    if not images_path.exists():
        return index
        
    print(f"Indexing images in {images_dir}...")
    for root, _, files in os.walk(images_path):
        for file in files:
            file_path = Path(root) / file
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    h = hashlib.sha256(content).hexdigest()
                    # Store relative path from images_dir's parent (split dir)
                    # e.g. if file is images/source/1.png, we want "images/source/1.png"
                    rel_path = file_path.relative_to(images_path.parent).as_posix()
                    index[h] = rel_path
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    # print(f"Indexed {len(index)} images.")
    return index

def restore_image_paths(dataset_dir):
    """
    Convert Base64 images in data.json back to relative file paths by matching content hash.
    """
    dataset_path = Path(dataset_dir)
    splits = ['train', 'test', 'valid']
    
    for split in splits:
        split_dir = dataset_path / split
        json_path = split_dir / "data.json"
        images_dir = split_dir / "images"
        
        if not json_path.exists():
            print(f"Skipping {split}: data.json not found.")
            continue
            
        print(f"Restoring paths for {split}...")
        
        # Build index of existing images
        image_index = build_image_index(images_dir)
        if not image_index:
            print(f"Warning: No images found in {images_dir}. Cannot restore paths.")
            continue
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        updated_count = 0
        
        for item in tqdm(data, desc=f"Processing {split}"):
            if 'images' in item and item['images']:
                new_images = []
                for img_entry in item['images']:
                    img_data = None
                    is_dict = False
                    
                    if isinstance(img_entry, dict) and 'image' in img_entry:
                        img_data = img_entry['image']
                        is_dict = True
                    elif isinstance(img_entry, str):
                        img_data = img_entry
                    
                    if img_data and len(img_data) > 500:
                        try:
                            # Decode and hash
                            if ',' in img_data:
                                img_data_clean = img_data.split(',', 1)[1]
                            else:
                                img_data_clean = img_data
                                
                            img_bytes = base64.b64decode(img_data_clean)
                            h = hashlib.sha256(img_bytes).hexdigest()
                            
                            if h in image_index:
                                rel_path = image_index[h]
                                if is_dict:
                                    new_images.append({"image": rel_path})
                                else:
                                    new_images.append(rel_path)
                            else:
                                print(f"Warning: Hash {h} not found in index for item {item.get('extra_info', {}).get('index')}")
                                new_images.append(img_entry)
                        except Exception as e:
                            print(f"Error decoding base64: {e}")
                            new_images.append(img_entry)
                    else:
                        # Already a path or short string
                        new_images.append(img_entry)
                
                if new_images != item['images']:
                    item['images'] = new_images
                    updated_count += 1
                    
        print(f"Updated {updated_count} items in {split}.")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    print("Restoration complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert dataset to veRL format or restore image paths")
    parser.add_argument("--dataset_dir", type=str, default=r"E:\套瓷三剑客\math-pro\dataset-verl", help="Path to the dataset directory")
    parser.add_argument("--mode", type=str, default="restore_paths", choices=["convert", "restore_paths"], help="Mode: 'convert' to Base64, 'restore_paths' to revert to relative paths")
    args = parser.parse_args()
    
    if args.mode == "convert":
        convert_dataset(args.dataset_dir)
    else:
        restore_image_paths(args.dataset_dir)
