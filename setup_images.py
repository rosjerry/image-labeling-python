#!/usr/bin/env python3
"""
Setup script to help organize images for training.
This script helps you understand where to place your images.
"""

import os
import json
import shutil
from pathlib import Path


def analyze_json_structure(json_path: str):
    """Analyze the JSON structure to understand image paths."""
    print("Analyzing JSON structure...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} samples in JSON file")
    
    # Analyze image paths
    image_paths = []
    for item in data[:5]:  # Check first 5 items
        if 'data' in item and 'image' in item['data']:
            image_path = item['data']['image']
            image_paths.append(image_path)
            print(f"Sample image path: {image_path}")
    
    return image_paths


def create_image_directory_structure():
    """Create a suggested directory structure for images."""
    print("\nCreating suggested directory structure...")
    
    # Create images directory
    images_dir = "images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Create subdirectories based on your JSON structure
    subdirs = ["toyota/rav4/45413425", "toyota/rav4/47932925", "toyota/rav4/51759035", 
               "toyota/rav4/53047455", "toyota/rav4/54572335"]
    
    for subdir in subdirs:
        full_path = os.path.join(images_dir, subdir)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")
    
    print(f"\nDirectory structure created in: {images_dir}")
    return images_dir


def provide_setup_instructions():
    """Provide clear setup instructions."""
    print("\n" + "="*60)
    print("IMAGE DIRECTORY SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1. ORGANIZE YOUR IMAGES:")
    print("   Based on your JSON data, you need to organize images like this:")
    print("   images/")
    print("   ├── toyota/rav4/45413425/")
    print("   │   ├── 45413425_Image_1.jpg")
    print("   │   ├── 45413425_Image_2.jpg")
    print("   │   └── ...")
    print("   ├── toyota/rav4/47932925/")
    print("   │   ├── 47932925_Image_1.jpg")
    print("   │   └── ...")
    print("   └── ...")
    
    print("\n2. ALTERNATIVE: FLAT STRUCTURE")
    print("   If you prefer a flat structure, you can:")
    print("   - Put all images in one directory (e.g., 'images/')")
    print("   - Update the data_loader.py to handle flat structure")
    
    print("\n3. TRAINING COMMANDS:")
    print("   Once images are organized, run:")
    print("   python main.py --json_path project-1-at-2025-10-05-21-43-98b8ac33.json \\")
    print("                   --image_dir images/ \\")
    print("                   --backbone resnet50 \\")
    print("                   --epochs 50")
    
    print("\n4. QUICK TEST:")
    print("   Test if everything works:")
    print("   python test_pipeline.py")


def check_current_setup():
    """Check current setup and provide recommendations."""
    print("Checking current setup...")
    
    # Check if JSON exists
    json_path = "project-1-at-2025-10-05-21-43-98b8ac33.json"
    if os.path.exists(json_path):
        print(f"✓ JSON file found: {json_path}")
        
        # Analyze JSON structure
        try:
            image_paths = analyze_json_structure(json_path)
        except Exception as e:
            print(f"❌ Error reading JSON: {e}")
            return
    else:
        print(f"❌ JSON file not found: {json_path}")
        return
    
    # Check for images directory
    images_dir = "images"
    if os.path.exists(images_dir):
        print(f"✓ Images directory found: {images_dir}")
        
        # Count images
        image_files = []
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        print(f"✓ Found {len(image_files)} image files")
        
        if len(image_files) == 0:
            print("⚠️  No image files found in the directory")
            print("   You need to copy your images to the images/ directory")
    else:
        print(f"❌ Images directory not found: {images_dir}")
        print("   Creating directory structure...")
        create_image_directory_structure()
    
    # Provide next steps
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Copy your car images to the 'images/' directory")
    print("2. Organize them according to the structure shown above")
    print("3. Run: python main.py --json_path project-1-at-2025-10-05-21-43-98b8ac33.json --image_dir images/")
    print("4. Or test first: python test_pipeline.py")


def main():
    """Main setup function."""
    print("Multi-Label Car Classification - Image Setup Helper")
    print("="*60)
    
    check_current_setup()
    provide_setup_instructions()


if __name__ == "__main__":
    main()
