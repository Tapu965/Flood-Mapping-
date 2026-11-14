import numpy as np
import cv2
from PIL import Image
import yaml
import os

class ChangeDetector:
    def __init__(self, config_path='config.yaml', model_path=None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(self.script_dir)
        
        with open(os.path.join(self.base_dir, config_path), 'r') as f:
            self.config = yaml.safe_load(f)
            
    def get_water_mask(self, image):
        """Pure color-based water detection"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Broad range to catch all water, even under clouds
        lower_bound = np.array([80, 40, 20])
        upper_bound = np.array([140, 255, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Fill small holes inside the water (like tiny boats or clouds over water)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def detect_change(self, before_image, after_image):
        # 1. Get water masks
        before_mask = self.get_water_mask(before_image)
        after_mask = self.get_water_mask(after_image)
        
        # 2. Find Contours (Outlines) of the BEFORE water
        before_contours, _ = cv2.findContours(before_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Create the PERMANENT WATER mask based ONLY on the before image's outline
        permanent_water = np.zeros_like(before_mask)
        for cnt in before_contours:
            if cv2.contourArea(cnt) > 100:
                # Fill the contour completely to make a solid shape
                cv2.drawContours(permanent_water, [cnt], -1, 255, -1)
                
        # 4. Create a Buffer Zone (Expand the permanent water by 25 pixels)
        # This accounts for the lake naturally rising a little bit
        buffer_size = 25
        kernel_buffer = np.ones((buffer_size, buffer_size), np.uint8)
        permanent_water_buffered = cv2.dilate(permanent_water, kernel_buffer, iterations=1)
        
        # 5. Find the TRUE FLOOD
        # We look at the AFTER water, and subtract the BUFFERED permanent water.
        # Anything left is water that spilled far onto the land.
        flood_mask = cv2.subtract(after_mask, permanent_water_buffered)
        
        # Clean up isolated dots
        kernel_clean = np.ones((5, 5), np.uint8)
        flood_mask = cv2.morphologyEx(flood_mask, cv2.MORPH_OPEN, kernel_clean)
        
        # Convert to 0-1 floats for the app
        new_flood_water = flood_mask.astype(np.float32) / 255.0
        permanent_water_float = permanent_water.astype(np.float32) / 255.0
        total_water_float = after_mask.astype(np.float32) / 255.0
        
        return new_flood_water, permanent_water_float, total_water_float
    
    def calculate_statistics(self, flood_mask, permanent_water, image_size):
        h, w = image_size
        flood_resized = cv2.resize(flood_mask, (w, h))
        permanent_resized = cv2.resize(permanent_water, (w, h))
        
        total_pixels = h * w
        flood_pixels = np.sum(flood_resized > 0.5)
        permanent_pixels = np.sum(permanent_resized > 0.5)
        
        flood_percentage = (flood_pixels / total_pixels) * 100
        permanent_percentage = (permanent_pixels / total_pixels) * 100
        
        reference_area_km2 = 100
        pixel_area_km2 = reference_area_km2 / total_pixels
        flood_area_km2 = flood_pixels * pixel_area_km2
        permanent_area_km2 = permanent_pixels * pixel_area_km2
        
        if flood_percentage < 5: severity = "Minor"
        elif flood_percentage < 15: severity = "Moderate"
        elif flood_percentage < 30: severity = "Severe"
        else: severity = "Catastrophic"
        
        return {
            'flood_percentage': round(flood_percentage, 2),
            'flood_area_km2': round(flood_area_km2, 2),
            'flood_pixels': int(flood_pixels),
            'permanent_water_percentage': round(permanent_percentage, 2),
            'permanent_water_area_km2': round(permanent_area_km2, 2),
            'permanent_water_pixels': int(permanent_pixels),
            'total_pixels': total_pixels,
            'severity': severity
        }
    
    def create_comparison_image(self, before_img, after_img, flood_mask, permanent_water):
        h, w = before_img.shape[:2]
        flood_resized = cv2.resize(flood_mask, (w, h))
        permanent_resized = cv2.resize(permanent_water, (w, h))
        
        # Flood overlay (RED)
        flood_overlay = after_img.copy()
        flood_overlay[flood_resized > 0.5] = [255, 0, 0]
        flood_vis = cv2.addWeighted(after_img, 0.5, flood_overlay, 0.5, 0)
        
        # Permanent water overlay (BLUE)
        perm_overlay = after_img.copy()
        perm_overlay[permanent_resized > 0.5] = [0, 0, 255]
        perm_vis = cv2.addWeighted(after_img, 0.5, perm_overlay, 0.5, 0)
        
        comparison = np.hstack([before_img, after_img, flood_vis, perm_vis])
        comparison = cv2.copyMakeBorder(comparison, 50, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])
        
        labels = ['BEFORE', 'AFTER', 'NEW FLOOD (Red)', 'PERMANENT WATER (Blue)']
        x_positions = [w//2 - 50, (w + w//2) - 50, (2*w + w//2) - 100, (3*w + w//2) - 120]
        
        for label, x in zip(labels, x_positions):
            cv2.putText(comparison, label, (x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        
        return comparison
    
    def analyze_from_paths(self, before_path, after_path):
        before_img = np.array(Image.open(before_path).convert('RGB'))
        after_img = np.array(Image.open(after_path).convert('RGB'))
        
        target_size = (512, 512)
        before_img = cv2.resize(before_img, target_size)
        after_img = cv2.resize(after_img, target_size)
        
        flood_mask, permanent_water, total_water = self.detect_change(before_img, after_img)
        stats = self.calculate_statistics(flood_mask, permanent_water, target_size)
        comparison = self.create_comparison_image(before_img, after_img, flood_mask, permanent_water)
        
        return {
            'before_image': before_img,
            'after_image': after_img,
            'flood_mask': flood_mask,
            'permanent_water': permanent_water,
            'total_water': total_water,
            'statistics': stats,
            'comparison_visualization': comparison
        }