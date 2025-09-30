import os
import json
import glob
from pathlib import Path

# in advanced defined directory of images (with 'pwd' command for example)
images_location = "~/unzipped-data/toyota/rav4/47932925"

# define labels: fuel_type, steer_wheel, year, color, model, make
make = "toyota"
model = "rav4"
year = "2014"
color = "white"
fuel_type = "petroleum(gas_in_us)"
steer_wheel = "left"

def generate_labelstudio_import_json(images_dir, output_file="labelstudio_import.json"):
    """
    Generate JSON file for Label Studio import with images from specified directory
    """
    # Expand the tilde in the path
    images_path = os.path.expanduser(images_dir)
    
    # Check if directory exists
    if not os.path.exists(images_path):
        print(f"Error: Directory {images_path} does not exist")
        return
    
    # Get directory name for naming (last part of path)
    dir_name = os.path.basename(images_path)
    
    # Find all image files in the directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_path, ext)))
        image_files.extend(glob.glob(os.path.join(images_path, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {images_path}")
        return
    
    # Generate JSON structure for Label Studio with pre-annotations
    tasks = []
    for image_file in sorted(image_files):
        # Get relative path from the mounted directory
        # Remove the leading part to match the container path structure
        relative_path = image_file.replace("/home/admin/unzipped-data/", "")
        
        # Create pre-annotation predictions using the defined variables
        predictions = []
        
        # Add each label as a prediction
        label_mappings = {
            "make": make,
            "model": model,
            "year": year,
            "color": color,
            "fuel_type": fuel_type,
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
    
    # Write JSON file
    with open(output_file, 'w') as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Generated {output_file} with {len(tasks)} image tasks")
    print(f"Directory name used for naming: {dir_name}")
    print(f"Images found: {len(image_files)}")
    
    # Show first few entries as preview
    if tasks:
        print("\nPreview of generated JSON:")
        print(json.dumps(tasks[:2], indent=2))

# generate json file which is used to upload on local docker compose labelstudio, use labelstudio_import.json as example, use directory name numbers for naming
if __name__ == "__main__":
    generate_labelstudio_import_json(images_location)