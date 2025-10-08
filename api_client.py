#!/usr/bin/env python3
"""
Client library for the Car Classification API.
Use this to integrate the model into other applications.
"""

import requests
import json
from PIL import Image
import io
from typing import Dict, Any, Optional

class CarClassificationClient:
    """Client for the Car Classification API."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def predict_from_file(self, image_path: str) -> Dict[str, Any]:
        """Predict car attributes from image file."""
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = self.session.post(f"{self.base_url}/predict", files=files)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f"API error: {response.status_code} - {response.text}"}
                
        except Exception as e:
            return {'error': f"Client error: {str(e)}"}
    
    def predict_from_pil(self, image: Image.Image) -> Dict[str, Any]:
        """Predict car attributes from PIL Image object."""
        try:
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            files = {'image': ('image.jpg', img_byte_arr, 'image/jpeg')}
            response = self.session.post(f"{self.base_url}/predict", files=files)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f"API error: {response.status_code} - {response.text}"}
                
        except Exception as e:
            return {'error': f"Client error: {str(e)}"}
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {'error': f"Health check failed: {str(e)}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            response = self.session.get(f"{self.base_url}/model_info")
            return response.json()
        except Exception as e:
            return {'error': f"Model info failed: {str(e)}"}

def main():
    """Example usage of the client."""
    # Initialize client
    client = CarClassificationClient()
    
    # Check if API is running
    health = client.health_check()
    print("🏥 Health Check:", health)
    
    if 'error' in health:
        print("❌ API is not running. Start it with: python production_app.py")
        return
    
    # Get model info
    model_info = client.get_model_info()
    print("🤖 Model Info:", model_info)
    
    # Example prediction (replace with your image path)
    image_path = "images/sample_car.jpg"  # Replace with actual image path
    
    if os.path.exists(image_path):
        print(f"\n🔍 Predicting: {image_path}")
        result = client.predict_from_file(image_path)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print("✅ Prediction Results:")
            for attr, info in result.items():
                print(f"  {attr}: {info['label']} ({info['confidence']:.2%} confidence)")
    else:
        print(f"⚠️  Image not found: {image_path}")
        print("Place a car image in the images/ directory and update the path.")

if __name__ == "__main__":
    import os
    main()
