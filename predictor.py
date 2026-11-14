import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import yaml
import os

class FloodPredictor:
    def __init__(self, config_path='config.yaml', model_path=None):
        # Get base directory
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(self.script_dir)
        
        # Load config
        config_full_path = os.path.join(self.base_dir, config_path)
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.image_size = tuple(self.config['data']['image_size'])
        self.model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            full_model_path = os.path.join(self.base_dir, model_path)
            if not os.path.exists(full_model_path):
                raise FileNotFoundError(f"Model not found: {full_model_path}")
            
            self.model = tf.keras.models.load_model(full_model_path, compile=False)
            print(f"Model loaded successfully from {full_model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def predict(self, image_path):
        """Predict flood mask for an image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.image_size)
        img_array = np.array(img) / 255.0
        
        # For demo, use same image for pre and post
        # In production, you'd use actual pre and post images
        X = np.concatenate([img_array, img_array], axis=-1)
        X = np.expand_dims(X, axis=0)
        
        # Predict
        prediction = self.model.predict(X, verbose=0)[0]
        
        # Convert to binary mask
        binary_mask = (prediction > 0.5).astype(np.uint8)
        
        return binary_mask, prediction
    
    def overlay(self, image_path, mask):
        """Overlay flood mask on original image"""
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        
        # Prepare mask
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        mask_resized = cv2.resize(mask.astype(np.float32), (256, 256))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # Create red overlay for flooded areas
        overlay = img.copy()
        overlay[mask_binary > 0] = [255, 0, 0]  # Red
        
        # Blend
        result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        return result
    
    def get_stats(self, mask):
        """Calculate flood statistics"""
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        total_pixels = mask.size
        flooded_pixels = np.sum(mask > 0.5)
        percentage = (flooded_pixels / total_pixels) * 100
        
        # Determine severity
        if percentage < 10:
            severity = "Minor"
        elif percentage < 30:
            severity = "Moderate"
        elif percentage < 50:
            severity = "Severe"
        else:
            severity = "Catastrophic"
        
        return {
            'percentage': float(percentage),
            'severity': severity,
            'flooded_pixels': int(flooded_pixels),
            'total_pixels': int(total_pixels)
        }