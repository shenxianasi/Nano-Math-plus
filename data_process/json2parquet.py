import pandas as pd
import json
import os
from pathlib import Path
import argparse

def convert_json_to_parquet(json_path, parquet_path):
    """
    Reads a JSON file and converts it to a Parquet file using PyArrow engine.
    """
    if not os.path.exists(json_path):
        print(f"Error: File not found at {json_path}")
        return False

    print(f"Loading JSON from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(data)
        
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Save to Parquet
        print(f"Saving Parquet to {parquet_path}...")
        df.to_parquet(parquet_path, engine='pyarrow')
        print("Conversion successful.")
        return True
        
    except Exception as e:
        print(f"Failed to convert {json_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert JSON datasets to Parquet format for veRL GRPO training")
    parser.add_argument("--base_dir", type=str, default=r"E:\套瓷三剑客\math-pro\dataset-verl", help="Base directory of the dataset")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    splits = ["train", "valid"]
    
    for split in splits:
        split_dir = base_dir / split
        json_file = split_dir / "data.json"
        parquet_file = split_dir / "data.parquet"
        
        print(f"\nProcessing {split} split...")
        convert_json_to_parquet(json_file, parquet_file)

if __name__ == "__main__":
    main()
