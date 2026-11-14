"""
Enhanced Predictor with Damage Assessment Capabilities
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import yaml
import os
import json
from datetime import datetime

class EnhancedFloodPredictor:
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
        
        # Damage assessment parameters
        self.damage_params = {
            'cost_per_hectare': 50000,
            'population_density': 1000,
            'building_density': 50,
            'agriculture_value': 20000,
            'infrastructure_cost': 1000000
        }
        
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
        original_size = img.size
        img = img.resize(self.image_size)
        img_array = np.array(img) / 255.0
        
        # For demo, use same image for pre and post
        X = np.concatenate([img_array, img_array], axis=-1)
        X = np.expand_dims(X, axis=0)
        
        # Predict
        prediction = self.model.predict(X, verbose=0)[0]
        
        # Convert to binary mask
        binary_mask = (prediction > 0.5).astype(np.uint8)
        
        return binary_mask, prediction, original_size
    
    def overlay(self, image_path, mask):
        """Overlay flood mask on original image"""
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        
        # Prepare mask
        if len(mask.shape) == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask
        
        mask_resized = cv2.resize(mask_2d.astype(np.float32), (512, 512))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # Create red overlay for flooded areas
        overlay = img.copy()
        overlay[mask_binary > 0] = [255, 0, 0]
        
        # Blend
        result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        return result
    
    def calculate_area(self, mask, image_size, reference_area_km2=100):
        """Calculate flooded area in square kilometers"""
        if len(mask.shape) == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask
        
        total_pixels = mask_2d.size
        flooded_pixels = np.sum(mask_2d > 0.5)
        
        pixel_area_km2 = reference_area_km2 / total_pixels
        flooded_area_km2 = flooded_pixels * pixel_area_km2
        
        return float(flooded_area_km2)
    
    def estimate_damage(self, mask, image_size, land_use_type='mixed'):
        """Estimate economic damage based on flooded area and land use"""
        flooded_area_km2 = self.calculate_area(mask, image_size)
        flooded_hectares = flooded_area_km2 * 100
        
        if land_use_type == 'urban':
            damage = flooded_hectares * self.damage_params['cost_per_hectare'] * 5
            buildings_affected = flooded_area_km2 * self.damage_params['building_density']
            population_affected = flooded_area_km2 * self.damage_params['population_density']
            
        elif land_use_type == 'agricultural':
            damage = flooded_hectares * self.damage_params['agriculture_value']
            buildings_affected = flooded_area_km2 * 5
            population_affected = flooded_area_km2 * 50
            
        else:  # mixed
            damage = flooded_hectares * self.damage_params['cost_per_hectare'] * 2
            buildings_affected = flooded_area_km2 * self.damage_params['building_density'] * 0.5
            population_affected = flooded_area_km2 * self.damage_params['population_density'] * 0.5
        
        return {
            'flooded_area_km2': round(flooded_area_km2, 2),
            'flooded_hectares': round(flooded_hectares, 2),
            'estimated_damage_usd': round(damage, 2),
            'buildings_affected': int(buildings_affected),
            'population_affected': int(population_affected),
            'damage_category': self._categorize_damage(damage)
        }
    
    def _categorize_damage(self, damage):
        """Categorize damage severity"""
        if damage < 100000:
            return "Minor"
        elif damage < 500000:
            return "Moderate"
        elif damage < 2000000:
            return "Severe"
        else:
            return "Catastrophic"
    
    def assess_infrastructure(self, mask):
        """Assess impact on infrastructure"""
        if len(mask.shape) == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask
        
        road_length_km = 10
        flooded_roads_km = road_length_km * (np.sum(mask_2d > 0.5) / mask_2d.size)
        
        return {
            'total_roads_km': road_length_km,
            'flooded_roads_km': round(flooded_roads_km, 2),
            'roads_affected_percent': round((flooded_roads_km / road_length_km * 100), 1),
            'bridges_affected': int(flooded_roads_km / 2)
        }
    
    def assess_agriculture(self, mask):
        """Assess agricultural impact"""
        if len(mask.shape) == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask
        
        flooded_pixels = np.sum(mask_2d > 0.5)
        total_pixels = mask_2d.size
        
        agricultural_ratio = 0.4
        agricultural_pixels = total_pixels * agricultural_ratio
        flooded_agriculture_pixels = min(flooded_pixels, agricultural_pixels)
        
        crop_damage_percent = (flooded_agriculture_pixels / agricultural_pixels) * 100 if agricultural_pixels > 0 else 0
        
        return {
            'agricultural_area_percent': agricultural_ratio * 100,
            'crop_damage_percent': round(crop_damage_percent, 1),
            'estimated_crop_loss_usd': round(flooded_pixels * 10, 2),
            'recovery_time_months': self._estimate_recovery_time(crop_damage_percent)
        }
    
    def _estimate_recovery_time(self, damage_percent):
        """Estimate recovery time based on damage percentage"""
        if damage_percent < 20:
            return 1
        elif damage_percent < 50:
            return 3
        elif damage_percent < 80:
            return 6
        else:
            return 12
    
    def _get_severity(self, percentage):
        """Get flood severity level"""
        if percentage < 10:
            return "Minor"
        elif percentage < 30:
            return "Moderate"
        elif percentage < 50:
            return "Severe"
        else:
            return "Catastrophic"
    
    def _generate_recommendations(self, flood_percent, damage, agriculture):
        """Generate action recommendations"""
        recommendations = []
        
        if flood_percent > 30:
            recommendations.append("Immediate evacuation recommended for affected areas")
        
        if damage['estimated_damage_usd'] > 1000000:
            recommendations.append("Major economic impact - disaster relief funds needed")
        
        if agriculture['crop_damage_percent'] > 50:
            recommendations.append("Severe crop loss - food security at risk")
        
        if damage['population_affected'] > 1000:
            recommendations.append("Large population affected - shelter required")
        
        if not recommendations:
            recommendations.append("Situation is manageable - monitor water levels")
        
        return recommendations
    
    def generate_report(self, mask, image_path, image_size, land_use_type='mixed'):
        """Generate comprehensive damage assessment report"""
        # Get basic stats
        if len(mask.shape) == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask
        
        total_pixels = mask_2d.size
        flooded_pixels = np.sum(mask_2d > 0.5)
        percentage = (flooded_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Damage assessment
        damage = self.estimate_damage(mask, image_size, land_use_type)
        infrastructure = self.assess_infrastructure(mask)
        agriculture = self.assess_agriculture(mask)
        
        # Generate report
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_analyzed': os.path.basename(image_path),
            'flood_statistics': {
                'coverage_percentage': round(percentage, 1),
                'flooded_pixels': int(flooded_pixels),
                'total_pixels': int(total_pixels),
                'severity': self._get_severity(percentage)
            },
            'damage_assessment': damage,
            'infrastructure_impact': infrastructure,
            'agricultural_impact': agriculture,
            'recommendations': self._generate_recommendations(percentage, damage, agriculture)
        }
        
        return report