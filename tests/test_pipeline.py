#!/usr/bin/env python3
"""
Test script to verify the training pipeline works correctly.
Creates dummy data and tests the complete pipeline.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil

from car_classifier.data_loader import create_data_loaders, CarDataset
from car_classifier.model import create_model, MultiLabelLoss
from car_classifier.trainer import ModelTrainer


def create_dummy_data(num_samples: int = 100, image_size: int = 224):
    """Create dummy data for testing."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    images_dir = os.path.join(temp_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Create dummy JSON data
    dummy_data = []
    makes = ['toyota', 'honda']
    models = ['rav4', 'accord']
    steer_wheels = ['left', 'right']
    
    for i in range(num_samples):
        # Create dummy image
        image = Image.fromarray(np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8))
        image_path = os.path.join(images_dir, f'dummy_image_{i}.jpg')
        image.save(image_path)
        
        # Create dummy annotation
        sample = {
            "id": i + 1,
            "annotations": [{
                "result": [
                    {
                        "type": "choices",
                        "value": {"choices": [np.random.choice(makes)]},
                        "from_name": "make"
                    },
                    {
                        "type": "choices",
                        "value": {"choices": [np.random.choice(models)]},
                        "from_name": "model"
                    },
                    {
                        "type": "choices",
                        "value": {"choices": [np.random.choice(steer_wheels)]},
                        "from_name": "steer_wheel"
                    }
                ]
            }],
            "data": {
                "image": f"/data/local-files/?d=dummy/dummy_image_{i}.jpg"
            }
        }
        dummy_data.append(sample)
    
    # Save JSON file
    json_path = os.path.join(temp_dir, 'dummy_data.json')
    with open(json_path, 'w') as f:
        json.dump(dummy_data, f)
    
    return temp_dir, json_path, images_dir


def test_data_loader():
    """Test data loader functionality."""
    print("Testing data loader...")
    
    # Create dummy data
    temp_dir, json_path, images_dir = create_dummy_data(num_samples=50)
    
    try:
        # Test dataset creation
        dataset = CarDataset(json_path, images_dir)
        print(f"  ✓ Dataset created with {len(dataset)} samples")
        
        # Test label mappings
        num_classes = dataset.get_num_classes()
        print(f"  ✓ Number of classes: {num_classes}")
        
        # Test data loaders
        train_loader, val_loader, num_classes = create_data_loaders(
            json_path, images_dir, batch_size=4, image_size=224, val_split=0.2
        )
        print(f"  ✓ Train loader: {len(train_loader)} batches")
        print(f"  ✓ Val loader: {len(val_loader)} batches")
        
        # Test batch loading
        for images, labels in train_loader:
            print(f"  ✓ Batch shape: {images.shape}")
            print(f"  ✓ Labels: {list(labels.keys())}")
            break
        
        print("  ✓ Data loader test passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_model():
    """Test model functionality."""
    print("\nTesting model...")
    
    # Create dummy model
    num_classes = {'make': 5, 'model': 5, 'steer_wheel': 2}
    model = create_model(backbone='resnet18', num_classes=num_classes)
    print(f"  ✓ Model created with backbone: {model.backbone_name}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    predictions = model(dummy_input)
    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Predictions shape: {[(k, v.shape) for k, v in predictions.items()]}")
    
    # Test loss function
    loss_fn = MultiLabelLoss()
    targets = {
        'make': torch.randint(0, num_classes['make'], (2,)),
        'model': torch.randint(0, num_classes['model'], (2,)),
        'steer_wheel': torch.randint(0, num_classes['steer_wheel'], (2,))
    }
    losses = loss_fn(predictions, targets)
    print(f"  ✓ Loss computation successful: {losses['total_loss'].item():.4f}")
    
    # Test parameter counting
    from model import count_parameters
    param_counts = count_parameters(model)
    print(f"  ✓ Total parameters: {param_counts['total_params']:,}")
    print(f"  ✓ Trainable parameters: {param_counts['trainable_params']:,}")
    
    print("  ✓ Model test passed!")


def test_trainer():
    """Test trainer functionality."""
    print("\nTesting trainer...")
    
    # Create dummy data
    temp_dir, json_path, images_dir = create_dummy_data(num_samples=20)
    
    try:
        # Create data loaders
        train_loader, val_loader, num_classes = create_data_loaders(
            json_path, images_dir, batch_size=4, image_size=224, val_split=0.3
        )
        
        # Create model
        model = create_model(backbone='resnet18', num_classes=num_classes)
        
        # Create trainer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = ModelTrainer(model, device, save_dir='test_checkpoints')
        print(f"  ✓ Trainer created on device: {device}")
        
        # Test single epoch training
        print("  ✓ Testing single epoch training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            learning_rate=1e-3,
            save_best=False
        )
        
        print(f"  ✓ Training completed")
        print(f"  ✓ Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"  ✓ Final val loss: {history['val_loss'][-1]:.4f}")
        
        print("  ✓ Trainer test passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        if os.path.exists('test_checkpoints'):
            shutil.rmtree('test_checkpoints')


def test_integration():
    """Test complete pipeline integration."""
    print("\nTesting complete pipeline integration...")
    
    # Create dummy data
    temp_dir, json_path, images_dir = create_dummy_data(num_samples=30)
    
    try:
        # Test data loading
        train_loader, val_loader, num_classes = create_data_loaders(
            json_path, images_dir, batch_size=4, image_size=224, val_split=0.2
        )
        print(f"  ✓ Data loaded: {len(train_loader)} train, {len(val_loader)} val batches")
        
        # Test model creation
        model = create_model(backbone='resnet18', num_classes=num_classes)
        print(f"  ✓ Model created with {num_classes} classes")
        
        # Test training setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = ModelTrainer(model, device, save_dir='integration_test')
        
        # Test short training
        print("  ✓ Running short training session...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            learning_rate=1e-3,
            save_best=True
        )
        
        # Verify training worked
        assert len(history['train_loss']) == 2, "Training history incomplete"
        assert len(history['val_loss']) == 2, "Validation history incomplete"
        
        print(f"  ✓ Training loss: {history['train_loss'][-1]:.4f}")
        print(f"  ✓ Validation loss: {history['val_loss'][-1]:.4f}")
        
        # Test model saving/loading
        checkpoint_path = os.path.join('integration_test', 'best_model.pth')
        if os.path.exists(checkpoint_path):
            print("  ✓ Model checkpoint saved")
        
        print("  ✓ Integration test passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        if os.path.exists('integration_test'):
            shutil.rmtree('integration_test')


def main():
    """Run all tests."""
    print("="*60)
    print("TESTING MULTI-LABEL CAR CLASSIFICATION PIPELINE")
    print("="*60)
    
    try:
        test_data_loader()
        test_model()
        test_trainer()
        test_integration()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe pipeline is ready for training with your data.")
        print("\nTo start training:")
        print("python main.py --json_path your_data.json --image_dir your_images/")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
