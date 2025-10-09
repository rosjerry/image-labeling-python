import os
import json
import glob
from pathlib import Path

images_location = "~/unzipped-data/honda/accord"

make = "honda"
model = "accord"
steer_wheel = "left"

def generate_labelstudio_import_json(images_dir, lot_number=None, output_file=None):
    """
    Generate JSON file for Label Studio import with images from specified directory
    """
    images_path = os.path.expanduser(images_dir)
    
    if not os.path.exists(images_path):
        print(f"Error: Directory {images_path} does not exist")
        return
    
    dir_name = os.path.basename(images_path)
    
    # Generate output filename if not provided
    if output_file is None:
        if lot_number:
            output_file = f"imports/ls_imp_{make}_{model}_{lot_number}.json"
        else:
            output_file = f"imports/ls_imp_{make}_{model}_{dir_name}.json"
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_path, ext)))
        image_files.extend(glob.glob(os.path.join(images_path, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {images_path}")
        return
    
    tasks = []
    for image_file in sorted(image_files):
        relative_path = image_file.replace("/home/admin/unzipped-data/", "")
        
        predictions = []
        
        label_mappings = {
            "make": make,
            "model": model,
            "steer_wheel": steer_wheel,
        }
        
        for field_name, value in label_mappings.items():
            if value:
                predictions.append({
                    "value": {"choices": [value]},
                    "from_name": field_name,
                    "to_name": "image",
                    "type": "choices",
                })
        
        task = {
            "data": {
                "image": f"/data/local-files/?d={relative_path}"
            },
            "predictions": [
                {
                    "result": predictions,
                    "score": 1.0,
                    "model_version": "pre_annotation_v1"
                }
            ]
        }
        tasks.append(task)
    
    with open(output_file, 'w') as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Generated {output_file} with {len(tasks)} image tasks")
    print(f"Directory name used for naming: {dir_name}")
    print(f"Images found: {len(image_files)}")
    
    if tasks:
        print("\nPreview of generated JSON:")
        print(json.dumps(tasks[:2], indent=2))

def scan_lot_directories(parent_dir):
    """
    Scan parent directory for lot number subdirectories
    """
    parent_path = os.path.expanduser(parent_dir)
    
    if not os.path.exists(parent_path):
        print(f"Error: Parent directory {parent_path} does not exist")
        return []
    
    lot_directories = []
    for item in os.listdir(parent_path):
        item_path = os.path.join(parent_path, item)
        if os.path.isdir(item_path):
            # Check if directory name looks like a lot number (numeric)
            if item.isdigit():
                lot_directories.append((item, item_path))
    
    return sorted(lot_directories, key=lambda x: int(x[0]))

def process_all_lot_directories(parent_dir):
    """
    Process all lot number directories in the parent directory
    """
    lot_directories = scan_lot_directories(parent_dir)
    
    if not lot_directories:
        print(f"No lot number directories found in {parent_dir}")
        return
    
    print(f"Found {len(lot_directories)} lot number directories:")
    for lot_number, lot_path in lot_directories:
        print(f"  - {lot_number}: {lot_path}")
    
    print(f"\nProcessing all directories...")
    
    total_processed = 0
    for lot_number, lot_path in lot_directories:
        print(f"\n--- Processing lot {lot_number} ---")
        generate_labelstudio_import_json(lot_path, lot_number=lot_number)
        total_processed += 1
    
    print(f"\n=== Summary ===")
    print(f"Total directories processed: {total_processed}")

if __name__ == "__main__":
    process_all_lot_directories(images_location)