# Data Directory Structure

This directory contains all data files used for model training and inference.

## Directory Structure

```
data/
├── annotations/        # Input: Label Studio JSON files
│   ├── *.json         # Annotation files from Label Studio
├── images/            # Input: Training/test images
│   └── (your image files)
├── predictions/       # Output: Model predictions
│   ├── *.json        # Prediction results
└── raw/              # Raw data before processing
    └── (unprocessed files)
```

## Usage

### Training
```bash
python examples/train.py \
    --json_path data/annotations/project-1-at-2025-10-05-21-43-98b8ac33.json \
    --image_dir data/images/
```

### Inference
```bash
python -m car_classifier.inference \
    --model_path checkpoints/best_model.pth \
    --label_mapping checkpoints/label_mapping.json \
    --image_dir data/images/ \
    --output_path data/predictions/results.json
```

## File Descriptions

### annotations/
- Input annotation files in Label Studio JSON format
- Contains labeled training data

### images/
- Training and test images
- Organized by source or lot number (optional)

### predictions/
- Model inference outputs
- JSON files with prediction results

### raw/
- Unprocessed data files
- Source data before preparation

