# Multi-Label Car Classification

A PyTorch training pipeline for multi-label car image classification using Label Studio annotations. The model predicts car make, model, and steering wheel position from images.

## Features

- **Multi-label Classification**: Predicts make, model, and steering wheel position simultaneously
- **Pre-trained Models**: Supports ResNet, EfficientNet, and DenseNet architectures
- **Data Augmentation**: Comprehensive augmentation pipeline for better generalization
- **Label Studio Integration**: Direct parsing of Label Studio JSON format
- **Flexible Architecture**: Easy to modify for different backbones and datasets
- **Training Monitoring**: Early stopping, learning rate scheduling, and progress tracking
- **Model Checkpointing**: Automatic saving of best models and training history

## Project Structure

```
image-labeling-python/
├── src/
│   └── car_classifier/          # Core package
│       ├── __init__.py
│       ├── model.py             # Model architectures
│       ├── trainer.py           # Training logic
│       ├── data_loader.py       # Dataset and data loading
│       └── inference.py         # Inference utilities
├── scripts/
│   ├── labelstudio/             # Label Studio integration
│   │   ├── create_labels.py    # Create labels in bulk
│   │   ├── update_labels.py    # Update labels in bulk
│   │   └── convert_predictions.py  # Convert predictions to annotations
│   └── data_preparation/        # Data preparation utilities
│       ├── copy_images.py       # Copy images from multiple sources
│       └── setup_images.py      # Setup image directory structure
├── app/                         # Production API
│   ├── production_app.py        # Flask API server
│   └── api_client.py            # API client library
├── tests/
│   └── test_pipeline.py         # Pipeline tests
├── examples/
│   ├── train.py                 # Training example (main script)
│   └── example_usage.py         # Usage examples
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── checkpoints/                 # Model checkpoints
├── images/                      # Training images
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
└── README.md                    # This file
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd image-labeling-python
```

2. **Install the package and dependencies**:
```bash
# Install in development mode (recommended for development)
pip install -e .

# Or install directly
pip install .
```

This will install the `car_classifier` package and all dependencies from `requirements.txt`.

3. **Prepare your data**:
   - Place your Label Studio JSON file in the project directory
   - Organize your images in a directory (e.g., `images/`)
   - Update the paths in `config.yaml` or use command line arguments

## Usage

### Training

#### Basic Training
```bash
# Using the train script from examples
python examples/train.py --json_path project-1-at-2025-10-05-21-43-98b8ac33.json \
                         --image_dir images/ \
                         --backbone resnet50 \
                         --epochs 50 \
                         --batch_size 32

# Or if installed, use the command-line tool
car-classifier-train --json_path project-1-at-2025-10-05-21-43-98b8ac33.json \
                     --image_dir images/ \
                     --backbone resnet50 \
                     --epochs 50 \
                     --batch_size 32
```

#### Advanced Training with Custom Settings
```bash
python examples/train.py --json_path project-1-at-2025-10-05-21-43-98b8ac33.json \
                         --image_dir images/ \
                         --backbone efficientnet_b0 \
                         --epochs 2 \
                         --batch_size 16 \
                         --learning_rate 1e-4 \
                         --weight_decay 1e-4 \
                         --val_split 0.2 \
                         --save_dir checkpoints \
                         --device cuda
```

#### Training with Frozen Backbone
```bash
python examples/train.py --json_path project-1-at-2025-10-05-21-43-98b8ac33.json \
                         --image_dir images/ \
                         --freeze_backbone \
                         --epochs 20 \
                         --learning_rate 1e-3
```

### Inference

#### Single Image Prediction
```bash
# Using Python
python -m car_classifier.inference --model_path checkpoints/best_model.pth \
                                   --label_mapping checkpoints/label_mapping.json \
                                   --image_path test_image.jpg

# Or if installed, use the command-line tool
car-classifier-inference --model_path checkpoints/best_model.pth \
                         --label_mapping checkpoints/label_mapping.json \
                         --image_path test_image.jpg
```

#### Batch Prediction
```bash
python -m car_classifier.inference --model_path checkpoints/best_model.pth \
                                   --label_mapping checkpoints/label_mapping.json \
                                   --image_dir test_images/ \
                                   --output_path predictions.json
```

## Configuration

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--json_path` | Path to Label Studio JSON file | Required |
| `--image_dir` | Directory containing images | Required |
| `--backbone` | Model architecture | `resnet50` |
| `--epochs` | Number of training epochs | `100` |
| `--batch_size` | Batch size | `32` |
| `--learning_rate` | Learning rate | `1e-4` |
| `--val_split` | Validation split ratio | `0.2` |
| `--save_dir` | Directory to save checkpoints | `checkpoints` |

### Supported Model Architectures

- **ResNet**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **EfficientNet**: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`
- **DenseNet**: `densenet121`, `densenet161`, `densenet169`, `densenet201`

## Data Format

The pipeline expects Label Studio JSON format with the following structure:

```json
[
  {
    "id": 2498,
    "annotations": [
      {
        "result": [
          {
            "type": "choices",
            "value": {"choices": ["toyota"]},
            "from_name": "make"
          },
          {
            "type": "choices", 
            "value": {"choices": ["rav4"]},
            "from_name": "model"
          },
          {
            "type": "choices",
            "value": {"choices": ["left"]},
            "from_name": "steer_wheel"
          }
        ]
      }
    ],
    "data": {
      "image": "/data/local-files/?d=toyota/rav4/45413425/45413425_Image_1.jpg"
    }
  }
]
```

## Model Architecture

The model consists of:

1. **Pre-trained Backbone**: Feature extractor (ResNet, EfficientNet, etc.)
2. **Multi-head Classifier**: Separate classification heads for each attribute
3. **Dropout Layers**: Regularization for better generalization

```
Input Image (3, 224, 224)
    ↓
Pre-trained Backbone
    ↓
Global Average Pooling
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│   Make Head     │   Model Head    │ Steer Wheel Head│
│   (Dropout +    │   (Dropout +    │   (Dropout +    │
│    FC + ReLU +  │    FC + ReLU +  │    FC + ReLU +  │
│    Dropout +    │    Dropout +    │    Dropout +    │
│    FC)          │    FC)          │    FC)          │
└─────────────────┴─────────────────┴─────────────────┘
```

## Training Features

### Data Augmentation
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transformations
- Normalization (ImageNet statistics)

### Training Optimizations
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Reduces learning rate over time
- **Model Checkpointing**: Saves best model automatically
- **Progress Tracking**: Real-time training metrics

### Validation
- Automatic train/validation split
- Comprehensive metrics for each attribute
- Best model selection based on validation loss

## Output Files

After training, the following files are created in the save directory:

- `best_model.pth`: Best model checkpoint
- `final_model.pth`: Final model checkpoint
- `training_history.json`: Training metrics
- `config.json`: Training configuration
- `training_curves.png`: Training curves plot (if matplotlib available)

## Performance Tips

### For Better Accuracy
1. **Use larger models**: ResNet101, EfficientNet-B2
2. **Increase training time**: More epochs with early stopping
3. **Data augmentation**: Enable all augmentation techniques
4. **Fine-tuning**: Start with frozen backbone, then unfreeze

### For Faster Training
1. **Use smaller models**: ResNet18, EfficientNet-B0
2. **Freeze backbone**: Train only classification heads
3. **Reduce image size**: 224x224 is usually sufficient
4. **Increase batch size**: If you have enough GPU memory

### For Limited Data
1. **Use pre-trained models**: Always use `--pretrained`
2. **Freeze backbone initially**: Train heads first, then unfreeze
3. **Strong augmentation**: Increase augmentation parameters
4. **Transfer learning**: Use models pre-trained on similar data

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size: `--batch_size 16`
   - Use smaller model: `--backbone resnet18`
   - Reduce image size: `--image_size 128`

2. **Slow Training**:
   - Use GPU: `--device cuda`
   - Increase workers: `--num_workers 8`
   - Use mixed precision training

3. **Poor Accuracy**:
   - Check data quality and labels
   - Increase training time: `--epochs 200`
   - Use data augmentation
   - Try different architectures

4. **Label Studio JSON Issues**:
   - Verify JSON format matches expected structure
   - Check image paths are correct
   - Ensure all required attributes are present

## Examples

### Quick Start
```bash
# 1. Install the package
pip install -e .

# 2. Train with default settings
python examples/train.py --json_path data.json --image_dir images/

# 3. Run inference
python -m car_classifier.inference --model_path checkpoints/best_model.pth \
                                   --label_mapping checkpoints/label_mapping.json \
                                   --image_path test.jpg

# 4. Run production API
python app/production_app.py
```

### Production Training
```bash
# High-quality training setup
python examples/train.py --json_path data.json --image_dir images/ \
                         --backbone efficientnet_b2 \
                         --epochs 200 \
                         --batch_size 16 \
                         --learning_rate 5e-5 \
                         --weight_decay 1e-4 \
                         --val_split 0.15 \
                         --save_dir production_models
```

### Using Utility Scripts

#### Label Studio Scripts
```bash
# Create labels in bulk for Label Studio
python scripts/labelstudio/create_labels.py

# Update existing labels in bulk
python scripts/labelstudio/update_labels.py

# Convert predictions to annotations
python scripts/labelstudio/convert_predictions.py
```

#### Data Preparation Scripts
```bash
# Copy images from multiple directories
python scripts/data_preparation/copy_images.py

# Setup image directory structure
python scripts/data_preparation/setup_images.py
```

### Running the Production API
```bash
# Start the Flask API server
python app/production_app.py

# The API will be available at http://localhost:5000
# You can test it with the client:
python app/api_client.py
```

### Using as a Python Package
After installation, you can import the package in your Python code:

```python
from car_classifier import create_model, ModelTrainer, create_data_loaders

# Load data
train_loader, val_loader, num_classes = create_data_loaders(
    json_path='data.json',
    image_dir='images/',
    batch_size=32
)

# Create model
model = create_model(backbone='resnet50', num_classes=num_classes)

# Train
trainer = ModelTrainer(model, device='cuda')
trainer.train(train_loader, val_loader, num_epochs=50)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the excellent framework
- Label Studio for annotation tools
- Timm library for model implementations
- The computer vision community for pre-trained models
