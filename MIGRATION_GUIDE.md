# Migration Guide: Project Refactoring

This guide helps you migrate to the new refactored project structure.

## What Changed?

The project has been reorganized following Python best practices. Files have been moved into purpose-based directories with a proper package structure.

## New Project Structure

```
image-labeling-python/
├── src/car_classifier/        # Core package (installable)
├── scripts/                   # Utility scripts
│   ├── labelstudio/          # Label Studio tools
│   └── data_preparation/     # Data prep tools
├── app/                       # Production API
├── tests/                     # Test suite
├── examples/                  # Example scripts
└── setup.py                   # Package installer
```

## File Mapping

### Core Files Moved to `src/car_classifier/`
- `model.py` → `src/car_classifier/model.py`
- `trainer.py` → `src/car_classifier/trainer.py`
- `data_loader.py` → `src/car_classifier/data_loader.py`
- `inference.py` → `src/car_classifier/inference.py`

### Scripts Moved to `scripts/`
- `create_label.py` → `scripts/labelstudio/create_labels.py`
- `update_label.py` → `scripts/labelstudio/update_labels.py`
- `convert.py` → `scripts/labelstudio/convert_predictions.py`
- `copy_images.py` → `scripts/data_preparation/copy_images.py`
- `setup_images.py` → `scripts/data_preparation/setup_images.py`

### Application Files Moved to `app/`
- `production_app.py` → `app/production_app.py`
- `api_client.py` → `app/api_client.py`

### Example/Test Files Moved
- `main.py` → `examples/train.py`
- `example_usage.py` → `examples/example_usage.py`
- `test_pipeline.py` → `tests/test_pipeline.py`

## How to Migrate

### 1. Install the Package

First, install the package in development mode:

```bash
pip install -e .
```

This makes the `car_classifier` package available throughout your environment.

### 2. Update Import Statements

**Old imports:**
```python
from model import create_model
from trainer import ModelTrainer
from data_loader import create_data_loaders
```

**New imports:**
```python
from car_classifier.model import create_model
from car_classifier.trainer import ModelTrainer
from car_classifier.data_loader import create_data_loaders
```

### 3. Update Script Paths

**Training:**
```bash
# Old
python main.py --json_path data.json --image_dir images/

# New
python examples/train.py --json_path data.json --image_dir images/

# Or use the command-line tool
car-classifier-train --json_path data.json --image_dir images/
```

**Inference:**
```bash
# Old
python inference.py --model_path checkpoints/best_model.pth --image_path test.jpg

# New
python -m car_classifier.inference --model_path checkpoints/best_model.pth --image_path test.jpg

# Or use the command-line tool
car-classifier-inference --model_path checkpoints/best_model.pth --image_path test.jpg
```

**Label Studio Scripts:**
```bash
# Old
python create_label.py
python update_label.py
python convert.py

# New
python scripts/labelstudio/create_labels.py
python scripts/labelstudio/update_labels.py
python scripts/labelstudio/convert_predictions.py
```

**Data Preparation:**
```bash
# Old
python copy_images.py
python setup_images.py

# New
python scripts/data_preparation/copy_images.py
python scripts/data_preparation/setup_images.py
```

**Production API:**
```bash
# Old
python production_app.py

# New
python app/production_app.py
```

### 4. Update Custom Scripts

If you have custom scripts that import from the old structure, update them:

**Before:**
```python
from model import create_model
from data_loader import CarDataset

model = create_model(num_classes={'make': 5})
dataset = CarDataset('data.json', 'images/')
```

**After:**
```python
from car_classifier.model import create_model
from car_classifier.data_loader import CarDataset

model = create_model(num_classes={'make': 5})
dataset = CarDataset('data.json', 'images/')
```

## Benefits of the New Structure

1. **Clean Organization**: Files are grouped by purpose
2. **Installable Package**: Can install with `pip install -e .`
3. **Better Imports**: Use standard Python package imports
4. **Professional Structure**: Follows Python best practices
5. **Easier Testing**: Clear separation of tests and examples
6. **Scalability**: Easy to add new modules

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'car_classifier'`:
```bash
# Make sure you've installed the package
pip install -e .
```

### Script Not Found

If scripts aren't found in the old location:
- Check the file mapping above
- Use the new paths listed in this guide

### Checkpoints/Data

All your existing data files remain unchanged:
- `checkpoints/` - Still in the same location
- `images/` - Still in the same location
- `*.json` annotation files - Still in the root directory

## Quick Reference

| Old Command | New Command |
|-------------|-------------|
| `python main.py ...` | `python examples/train.py ...` |
| `python inference.py ...` | `python -m car_classifier.inference ...` |
| `python production_app.py` | `python app/production_app.py` |
| `python create_label.py` | `python scripts/labelstudio/create_labels.py` |
| `python copy_images.py` | `python scripts/data_preparation/copy_images.py` |

## Need Help?

If you encounter any issues with the migration:
1. Check this guide for the correct paths
2. Ensure the package is installed (`pip install -e .`)
3. Verify your Python environment is activated
4. Check the README.md for updated examples

## Rollback (if needed)

The old files have been removed, but you can find them in your git history:
```bash
git log --all --full-history -- model.py
git checkout <commit-hash> -- model.py
```

