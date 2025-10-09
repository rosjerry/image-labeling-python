#!/usr/bin/env python3
"""
Production-ready car classification web application.
Uses your trained model to classify car images in real-time.
"""

import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
import io
import base64
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import logging

from car_classifier.model import create_model
from car_classifier.data_loader import get_data_transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model
model = None
label_mappings = None
device = None

def load_model():
    """Load the trained model and label mappings."""
    global model, label_mappings, device
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model checkpoint
        checkpoint_path = 'checkpoints/best_model.pth'
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Get model configuration
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            model = create_model(**model_config)
        else:
            # Use the correct class counts from the trained model
            num_classes = {'make': 2, 'model': 2, 'steer_wheel': 1}
            model = create_model(num_classes=num_classes)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Load label mappings
        label_mapping_path = 'checkpoints/label_mapping.json'
        logger.info(f"Looking for label mappings at: {os.path.abspath(label_mapping_path)}")
        if os.path.exists(label_mapping_path):
            with open(label_mapping_path, 'r') as f:
                label_mappings = json.load(f)
            logger.info(f"Loaded label mappings: {label_mappings}")
        else:
            # Create default mappings if not found
            label_mappings = {
                'make': {'honda': 0, 'toyota': 1},
                'model': {'accord': 0, 'rav4': 1},
                'steer_wheel': {'left': 0}
            }
            logger.warning("Label mappings not found, using default mappings")
        
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def preprocess_image(image):
    """Preprocess image for model inference."""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get transforms
        _, transform = get_data_transforms()
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        return image_tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def predict_image(image):
    """Predict car attributes from image."""
    global model, label_mappings
    
    try:
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Get predictions
        with torch.no_grad():
            predictions = model(image_tensor)
        
        # Convert predictions to labels and probabilities
        results = {}
        for attr_name, logits in predictions.items():
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            # Convert to label name
            if attr_name in label_mappings:
                reverse_mapping = {v: k for k, v in label_mappings[attr_name].items()}
                label_name = reverse_mapping.get(predicted_class, f"class_{predicted_class}")
            else:
                label_name = f"class_{predicted_class}"
            
            results[attr_name] = {
                'label': label_name,
                'confidence': float(confidence),
                'class_id': predicted_class
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Classification API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            cursor: pointer;
            transition: border-color 0.3s;
        }
        .upload-area:hover {
            border-color: #007bff;
        }
        .upload-area.dragover {
            border-color: #007bff;
            background-color: #f8f9fa;
        }
        input[type="file"] {
            display: none;
        }
        .btn {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .attribute {
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .confidence {
            font-weight: bold;
            color: #007bff;
        }
        .error {
            color: #dc3545;
            background-color: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .loading {
            text-align: center;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 Car Classification API</h1>
        <p>Upload a car image to classify its make and model</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <p>📁 Click here or drag and drop an image</p>
            <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
        </div>
        
        <div id="result"></div>
    </div>

    <script>
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                uploadImage(file);
            }
        }

        function uploadImage(file) {
            const formData = new FormData();
            formData.append('image', file);
            
            document.getElementById('result').innerHTML = '<div class="loading">🔄 Processing image...</div>';
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('result').innerHTML = 
                        `<div class="error">❌ Error: ${data.error}</div>`;
                } else {
                    displayResults(data);
                }
            })
            .catch(error => {
                document.getElementById('result').innerHTML = 
                    `<div class="error">❌ Error: ${error.message}</div>`;
            });
        }

        function displayResults(data) {
            let html = '<div class="result"><h3>🎯 Classification Results</h3>';
            
            for (const [attribute, info] of Object.entries(data)) {
                // Exclude steering wheel position from display
                if (attribute === 'steer_wheel') {
                    continue;
                }
                
                html += `
                    <div class="attribute">
                        <strong>${attribute.replace('_', ' ').toUpperCase()}:</strong> 
                        ${info.label} 
                    </div>
                `;
            }
            
            html += '</div>';
            document.getElementById('result').innerHTML = html;
        }

        // Drag and drop functionality
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                uploadImage(files[0]);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction."""
    try:
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read and process image
        image = Image.open(file.stream)
        
        # Get predictions
        predictions = predict_image(image)
        
        return jsonify(predictions)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'unknown'
    })

@app.route('/model_info')
def model_info():
    """Get model information."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_loaded': True,
        'device': str(device),
        'label_mappings': label_mappings,
        'model_parameters': sum(p.numel() for p in model.parameters())
    })

if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        logger.error("Failed to load model. Exiting.")
        exit(1)
    
    # Start the web server
    logger.info("Starting Car Classification API server...")
    logger.info("Open http://localhost:5000 in your browser")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
