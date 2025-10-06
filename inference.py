#!/usr/bin/env python3
"""
Inference script for multi-label car classification.
Loads a trained model and performs inference on new images.
"""

import argparse
import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
import glob

from model import create_model
from data_loader import get_data_transforms


class CarClassifierInference:
    """
    Inference class for multi-label car classification.
    """
    
    def __init__(self, model_path: str, label_mapping_path: str, device: str = 'auto'):
        """
        Initialize inference class.
        
        Args:
            model_path: Path to trained model checkpoint
            label_mapping_path: Path to label mapping JSON file
            device: Device to use for inference
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        
        # Load label mappings
        with open(label_mapping_path, 'r') as f:
            self.label_mappings = json.load(f)
        
        # Create reverse mappings
        self.reverse_mappings = {}
        for attr_name, mapping in self.label_mappings.items():
            self.reverse_mappings[attr_name] = {v: k for k, v in mapping.items()}
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Get transforms
        _, self.transform = get_data_transforms()
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model configuration from checkpoint
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            model = create_model(**model_config)
        else:
            # Fallback: create model with default config
            num_classes = {attr_name: len(mapping) 
            for attr_name, mapping in self.label_mappings.items()}
            model = create_model(num_classes=num_classes)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def predict_single(self, image_path: str) -> Dict[str, Tuple[str, float]]:
        """
        Predict labels for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with predictions for each attribute
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Convert predictions to labels and probabilities
        results = {}
        for attr_name, logits in predictions.items():
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            # Convert to label name
            if attr_name in self.reverse_mappings:
                label_name = self.reverse_mappings[attr_name][predicted_class]
            else:
                label_name = str(predicted_class)
            
            results[attr_name] = (label_name, confidence)
        
        return results
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Tuple[str, float]]]:
        """
        Predict labels for a batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({})
        
        return results
    
    def predict_directory(self, image_dir: str, extensions: List[str] = None) -> Dict[str, Dict[str, Tuple[str, float]]]:
        """
        Predict labels for all images in a directory.
        
        Args:
            image_dir: Directory containing images
            extensions: List of file extensions to process
            
        Returns:
            Dictionary mapping image paths to predictions
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Find all image files
        image_paths = []
        for ext in extensions:
            pattern = os.path.join(image_dir, f'*{ext}')
            image_paths.extend(glob.glob(pattern))
            pattern = os.path.join(image_dir, f'*{ext.upper()}')
            image_paths.extend(glob.glob(pattern))
        
        print(f"Found {len(image_paths)} images in {image_dir}")
        
        # Predict for all images
        results = {}
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                results[image_path] = result
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results[image_path] = {}
        
        return results


def save_predictions(predictions: Dict, output_path: str):
    """Save predictions to JSON file."""
    # Convert tuples to lists for JSON serialization
    serializable_predictions = {}
    for image_path, pred_dict in predictions.items():
        serializable_pred = {}
        for attr_name, (label, confidence) in pred_dict.items():
            serializable_pred[attr_name] = {
                'label': label,
                'confidence': confidence
            }
        serializable_predictions[image_path] = serializable_pred
    
    with open(output_path, 'w') as f:
        json.dump(serializable_predictions, f, indent=2)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference on car images')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--label_mapping', type=str, required=True,
                       help='Path to label mapping JSON file')
    parser.add_argument('--image_path', type=str,
                       help='Path to single image file')
    parser.add_argument('--image_dir', type=str,
                       help='Directory containing images')
    parser.add_argument('--output_path', type=str, default='predictions.json',
                       help='Path to save predictions (default: predictions.json)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_path and not args.image_dir:
        print("Error: Either --image_path or --image_dir must be specified")
        return
    
    if args.image_path and args.image_dir:
        print("Error: Cannot specify both --image_path and --image_dir")
        return
    
    # Create inference object
    print("Loading model...")
    classifier = CarClassifierInference(
        model_path=args.model_path,
        label_mapping_path=args.label_mapping,
        device=args.device
    )
    
    # Run inference
    if args.image_path:
        print(f"Processing single image: {args.image_path}")
        predictions = classifier.predict_single(args.image_path)
        
        print("\nPredictions:")
        for attr_name, (label, confidence) in predictions.items():
            print(f"  {attr_name}: {label} (confidence: {confidence:.4f})")
        
        # Save single prediction
        single_pred = {args.image_path: predictions}
        save_predictions(single_pred, args.output_path)
        
    else:
        print(f"Processing directory: {args.image_dir}")
        predictions = classifier.predict_directory(args.image_dir)
        
        print(f"\nProcessed {len(predictions)} images")
        
        # Show sample predictions
        sample_count = min(5, len(predictions))
        print(f"\nSample predictions (first {sample_count}):")
        for i, (image_path, pred_dict) in enumerate(list(predictions.items())[:sample_count]):
            print(f"\n{i+1}. {os.path.basename(image_path)}:")
            for attr_name, (label, confidence) in pred_dict.items():
                print(f"   {attr_name}: {label} (confidence: {confidence:.4f})")
        
        # Save all predictions
        save_predictions(predictions, args.output_path)
    
    print(f"\nPredictions saved to: {args.output_path}")


if __name__ == "__main__":
    main()
