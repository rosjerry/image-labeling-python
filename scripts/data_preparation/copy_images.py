#!/usr/bin/env python3
"""
Script to copy all images from multiple source directories to a single destination.
Copies images while preserving exact filenames.
"""

import os
import shutil
from pathlib import Path


def copy_images_recursive(source_dirs, destination_dir):
    """
    Copy all images from source directories to destination directory.

    Args:
        source_dirs: List of source directories to search
        destination_dir: Destination directory to copy images to
    """
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    copied_count = 0
    skipped_count = 0

    print(f"Copying images to: {destination_dir}")
    print("=" * 50)

    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"⚠️  Source directory not found: {source_dir}")
            continue

        print(f"\n📁 Processing: {source_dir}")

        # Walk through all subdirectories
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                # Check if file is an image
                file_ext = Path(file).suffix.lower()
                if file_ext in image_extensions:
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(destination_dir, file)

                    try:
                        # Check if file already exists in destination
                        if os.path.exists(destination_path):
                            print(f"⚠️  File already exists, skipping: {file}")
                            skipped_count += 1
                        else:
                            # Copy the file
                            shutil.copy2(source_path, destination_path)
                            print(f"✓ Copied: {file}")
                            copied_count += 1

                    except Exception as e:
                        print(f"❌ Error copying {file}: {e}")
                        skipped_count += 1

    print("\n" + "=" * 50)
    print(f"📊 Summary:")
    print(f"   ✓ Copied: {copied_count} images")
    print(f"   ⚠️  Skipped: {skipped_count} images")
    print(f"   📁 Destination: {destination_dir}")


def main():
    """Main function to copy images."""
    # Source directories (where your images are currently located)
    source_directories = [
        "/home/admin/unzipped-data/toyota/rav4",
        "/home/admin/unzipped-data/honda/accord",
    ]

    # Destination directory (where you want all images)
    destination_directory = "/home/admin/works/image-labeling-python/images"

    print("🖼️  Image Copy Script")
    print("=" * 50)
    print(f"Source directories:")
    for i, src in enumerate(source_directories, 1):
        print(f"  {i}. {src}")
    print(f"\nDestination: {destination_directory}")

    # Confirm before proceeding
    response = input("\nProceed with copying? (y/n): ").lower().strip()
    if response not in ["y", "yes"]:
        print("❌ Operation cancelled.")
        return

    # Copy images
    copy_images_recursive(source_directories, destination_directory)

    print(f"\n🎉 Done! All images copied to: {destination_directory}")
    print("\nNext steps:")
    print("1. Verify images are in the destination folder")
    print(
        "2. Run training: python main.py --json_path project-1-at-2025-10-05-21-43-98b8ac33.json --image_dir images/"
    )


if __name__ == "__main__":
    main()
