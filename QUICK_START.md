# Quick Start Guide - Refactored Project

## 🚀 Getting Started (5 minutes)

### 1. Install the Package
```bash
cd /home/rostomi/works/image-labeling-python
pip install -e .
```

### 2. Run Training
```bash
python examples/train.py \
    --json_path data/annotations/project-1-at-2025-10-05-21-43-98b8ac33.json \
    --image_dir data/images/ \
    --epochs 50 \
    --batch_size 32
```

### 3. Run Inference
```bash
python -m car_classifier.inference \
    --model_path checkpoints/best_model.pth \
    --label_mapping checkpoints/label_mapping.json \
    --image_path your_test_image.jpg
```

### 4. Start Production API
```bash
python app/production_app.py
# Access at http://localhost:5000
```

## 📁 Where to Find Things

### Training & Inference
- **Training Script:** `examples/train.py`
- **Inference:** `python -m car_classifier.inference`
- **Model Code:** `src/car_classifier/model.py`
- **Trainer:** `src/car_classifier/trainer.py`

### Label Studio Tools
- **Create Labels:** `python scripts/labelstudio/create_labels.py`
- **Update Labels:** `python scripts/labelstudio/update_labels.py`
- **Convert Predictions:** `python scripts/labelstudio/convert_predictions.py`

### Data Preparation
- **Copy Images:** `python scripts/data_preparation/copy_images.py`
- **Setup Images:** `python scripts/data_preparation/setup_images.py`

### Production
- **API Server:** `python app/production_app.py`
- **API Client:** `python app/api_client.py`

### Testing
- **Test Pipeline:** `python tests/test_pipeline.py`
- **Verify Structure:** `python3 verify_structure.py`

## 💡 Common Tasks

### Import the Package in Python
```python
from car_classifier import create_model, ModelTrainer, create_data_loaders

# Your code here
```

### Train a Model
```bash
python examples/train.py \
    --json_path data/annotations/training_data.json \
    --image_dir data/images/ \
    --backbone resnet50 \
    --epochs 100
```

### Run Inference on a Directory
```bash
python -m car_classifier.inference \
    --model_path checkpoints/best_model.pth \
    --label_mapping checkpoints/label_mapping.json \
    --image_dir data/images/ \
    --output_path data/predictions/results.json
```

### Use the API
```bash
# Terminal 1: Start server
python app/production_app.py

# Terminal 2: Test with client
python app/api_client.py
```

## 📚 Documentation

- **README.md** - Full documentation
- **MIGRATION_GUIDE.md** - Migrating from old structure
- **REFACTORING_SUMMARY.md** - What changed
- **QUICK_START.md** - This file

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'car_classifier'"
```bash
pip install -e .
```

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install -r requirements.txt
```

### Script not found
- Check the new paths in this guide
- Old files were moved/renamed

### Import errors in scripts
- Make sure package is installed: `pip install -e .`
- Check you're using the right Python environment

## ✅ Verify Everything Works

```bash
# 1. Check structure
python3 verify_structure.py

# 2. Test the pipeline
python tests/test_pipeline.py

# 3. Try training (quick test)
python examples/train.py --help
```

## 🎯 What Changed?

### Old Way:
```bash
python main.py --json_path data.json --image_dir images/
python inference.py --model_path model.pth --image_path test.jpg
python create_label.py
```

### New Way:
```bash
python examples/train.py --json_path data/annotations/data.json --image_dir data/images/
python -m car_classifier.inference --model_path model.pth --image_path data/images/test.jpg
python scripts/labelstudio/create_labels.py
```

## 📦 Package Structure

```
car_classifier/              # Installable package
├── model.py                # Model architectures
├── trainer.py              # Training logic  
├── data_loader.py          # Data loading
└── inference.py            # Inference tools
```

Import with:
```python
from car_classifier import create_model, ModelTrainer
from car_classifier.inference import CarClassifierInference
```

## 🆘 Need Help?

1. Read **MIGRATION_GUIDE.md** for detailed migration steps
2. Check **README.md** for full documentation
3. Run `python3 verify_structure.py` to check setup
4. Make sure virtual environment is activated

---

**Quick Reference:**
- Train: `python examples/train.py`
- Infer: `python -m car_classifier.inference`
- API: `python app/production_app.py`
- Label Studio: `python scripts/labelstudio/<script>.py`
- Data Prep: `python scripts/data_preparation/<script>.py`

