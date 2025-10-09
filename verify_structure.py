#!/usr/bin/env python3
"""
Verification script to test the refactored project structure.
Run this after refactoring to ensure everything works correctly.
"""

import sys
import os
from pathlib import Path

def check_directories():
    """Check that all expected directories exist."""
    print("🔍 Checking directory structure...")
    
    expected_dirs = [
        'src/car_classifier',
        'scripts/labelstudio',
        'scripts/data_preparation',
        'app',
        'tests',
        'examples',
    ]
    
    all_exist = True
    for directory in expected_dirs:
        path = Path(directory)
        if path.exists():
            print(f"  ✓ {directory}")
        else:
            print(f"  ✗ {directory} - NOT FOUND")
            all_exist = False
    
    return all_exist

def check_files():
    """Check that all expected files exist."""
    print("\n🔍 Checking key files...")
    
    expected_files = [
        'setup.py',
        'src/car_classifier/__init__.py',
        'src/car_classifier/model.py',
        'src/car_classifier/trainer.py',
        'src/car_classifier/data_loader.py',
        'src/car_classifier/inference.py',
        'scripts/labelstudio/create_labels.py',
        'scripts/labelstudio/update_labels.py',
        'scripts/labelstudio/convert_predictions.py',
        'scripts/data_preparation/copy_images.py',
        'scripts/data_preparation/setup_images.py',
        'app/production_app.py',
        'app/api_client.py',
        'tests/test_pipeline.py',
        'examples/train.py',
        'examples/example_usage.py',
    ]
    
    all_exist = True
    for file_path in expected_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def check_old_files_removed():
    """Check that old files have been removed from root."""
    print("\n🔍 Checking old files are removed...")
    
    old_files = [
        'model.py',
        'trainer.py',
        'data_loader.py',
        'inference.py',
        'create_label.py',
        'update_label.py',
        'convert.py',
        'copy_images.py',
        'setup_images.py',
        'production_app.py',
        'api_client.py',
        'test_pipeline.py',
        'example_usage.py',
        'main.py',
    ]
    
    all_removed = True
    for file_path in old_files:
        path = Path(file_path)
        if not path.exists():
            print(f"  ✓ {file_path} - removed")
        else:
            print(f"  ⚠️  {file_path} - still exists")
            all_removed = False
    
    return all_removed

def test_imports():
    """Test that the package can be imported."""
    print("\n🔍 Testing package imports...")
    
    try:
        # Add src to path for testing before installation
        sys.path.insert(0, str(Path(__file__).parent / 'src'))
        
        from car_classifier import create_model, ModelTrainer, create_data_loaders
        print("  ✓ Successfully imported: create_model, ModelTrainer, create_data_loaders")
        
        from car_classifier.model import MultiLabelCarClassifier, MultiLabelLoss
        print("  ✓ Successfully imported: MultiLabelCarClassifier, MultiLabelLoss")
        
        from car_classifier.data_loader import CarDataset, get_data_transforms
        print("  ✓ Successfully imported: CarDataset, get_data_transforms")
        
        from car_classifier.inference import CarClassifierInference
        print("  ✓ Successfully imported: CarClassifierInference")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        print("  💡 Try running: pip install -e .")
        return False

def main():
    """Run all verification checks."""
    print("="*60)
    print("🚀 Project Structure Verification")
    print("="*60)
    
    results = {
        'directories': check_directories(),
        'files': check_files(),
        'old_files_removed': check_old_files_removed(),
        'imports': test_imports(),
    }
    
    print("\n" + "="*60)
    print("📊 Verification Summary")
    print("="*60)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name.replace('_', ' ').title()}")
    
    if all(results.values()):
        print("\n🎉 All checks passed! The refactoring is complete.")
        print("\n📝 Next steps:")
        print("  1. Install the package: pip install -e .")
        print("  2. Read MIGRATION_GUIDE.md for usage changes")
        print("  3. Update any custom scripts with new imports")
        print("  4. Test your workflows with the new structure")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

