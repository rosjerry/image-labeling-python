# Data Organization Guide

## New Data Structure

All data files are now organized in the `data/` directory:

```
data/
├── annotations/        # INPUT: Label Studio JSON files
├── images/            # INPUT: Training/test images  
├── predictions/       # OUTPUT: Model predictions
└── raw/              # Raw data before processing
```

## Purpose of Each Directory

### 📥 Input Directories

#### `data/annotations/`
- **Purpose**: Store Label Studio annotation JSON files
- **Contents**: Training labels and annotations
- **Example files**:
  - `project-1-at-2025-10-05-21-43-98b8ac33.json`
  - `annotations_cars_train_1.json`
  - `labelstudio_import.json`

#### `data/images/`
- **Purpose**: Store training and test images
- **Organization**: Can be flat or organized by make/model/lot
- **Example structure**:
  ```
  data/images/
  ├── toyota/
  │   └── rav4/
  │       ├── image1.jpg
  │       └── image2.jpg
  └── honda/
      └── accord/
          ├── image1.jpg
          └── image2.jpg
  ```

#### `data/raw/`
- **Purpose**: Unprocessed source data
- **Use case**: Store original data before any transformations
- **Contents**: Downloaded files, exports, etc.

### 📤 Output Directories

#### `data/predictions/`
- **Purpose**: Store model inference results
- **Format**: JSON files with predictions and confidence scores
- **Example output**:
  ```json
  {
    "image_path": {
      "make": {"label": "toyota", "confidence": 0.95},
      "model": {"label": "rav4", "confidence": 0.89}
    }
  }
  ```

## Usage Examples

### Training
```bash
# Basic training
python examples/train.py \
    --json_path data/annotations/project-1-at-2025-10-05-21-43-98b8ac33.json \
    --image_dir data/images/ \
    --epochs 50

# Advanced training with all paths
python examples/train.py \
    --json_path data/annotations/annotations_cars_train_1.json \
    --image_dir data/images/ \
    --save_dir checkpoints/ \
    --epochs 100 \
    --batch_size 32
```

### Inference
```bash
# Single image prediction
python -m car_classifier.inference \
    --model_path checkpoints/best_model.pth \
    --label_mapping checkpoints/label_mapping.json \
    --image_path data/images/test_car.jpg

# Batch prediction on directory
python -m car_classifier.inference \
    --model_path checkpoints/best_model.pth \
    --label_mapping checkpoints/label_mapping.json \
    --image_dir data/images/ \
    --output_path data/predictions/batch_results.json
```

### Data Preparation Scripts
```bash
# Copy images from source directories
python scripts/data_preparation/copy_images.py
# (Update source paths in the script to point to your data)

# Setup image directory structure
python scripts/data_preparation/setup_images.py
```

## Migration from Old Structure

### Old locations → New locations

| Old Location | New Location | Purpose |
|--------------|--------------|---------|
| `images/` | `data/images/` | Training images |
| `*.json` (root) | `data/annotations/*.json` | Annotation files |
| `predictions.json` | `data/predictions/*.json` | Model outputs |

### Moving Your Data

If you have existing data, move it:

```bash
# Move images
mv images/* data/images/

# Move annotation files
mv *.json data/annotations/
# (except config files)

# Create symbolic link if needed for backward compatibility
ln -s data/images images
```

## Benefits of This Structure

1. ✅ **Clear separation** of input and output data
2. ✅ **Easy to backup** - just backup the `data/` directory
3. ✅ **Git-friendly** - easy to ignore large files
4. ✅ **Organized workflow** - raw → processed → predictions
5. ✅ **Professional structure** - follows data science best practices

## .gitignore Configuration

The following files are ignored by git:
- `data/images/` - Large image files
- `data/raw/` - Raw unprocessed data
- `data/annotations/*.json` - Annotation files (usually large)
- `data/predictions/*.json` - Prediction outputs

Keep in git:
- `data/README.md` - Directory documentation
- Directory structure (empty directories with `.gitkeep`)

## Tips

1. **Backup regularly**: The `data/` directory contains all your work
2. **Use descriptive names**: Name annotation files with dates/versions
3. **Organize predictions**: Create subdirectories by date or experiment
4. **Keep raw data**: Don't delete `data/raw/` - it's your source of truth
5. **Version annotations**: Use git for tracking annotation file versions

## Example Workflow

```bash
# 1. Place raw images
cp -r /source/images/* data/raw/

# 2. Prepare images for training
python scripts/data_preparation/copy_images.py

# 3. Create Label Studio annotations
python scripts/labelstudio/create_labels.py

# 4. Export annotations from Label Studio
# Save to: data/annotations/project-YYYY-MM-DD.json

# 5. Train model
python examples/train.py \
    --json_path data/annotations/project-2025-10-09.json \
    --image_dir data/images/

# 6. Run predictions
python -m car_classifier.inference \
    --model_path checkpoints/best_model.pth \
    --label_mapping checkpoints/label_mapping.json \
    --image_dir data/images/ \
    --output_path data/predictions/results-2025-10-09.json
```

