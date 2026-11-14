"""
TESTING SCRIPT - RUN AFTER 02_train.py
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.predictor import FloodPredictor

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("="*60)
print("STEP 2: TESTING FLOOD DETECTION MODEL")
print("="*60)

# Load model
model_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'flood_model.h5')
if not os.path.exists(model_path):
    print(f"Model not found at: {model_path}")
    print("Please run 02_train.py first")
    sys.exit(1)

predictor = FloodPredictor('config.yaml', 'models/trained_models/flood_model.h5')

# Test on images
print("\nTesting on sample images...")
data_path = os.path.join(BASE_DIR, 'data', 'raw')

# Get list of test images
test_images = []
for i in range(30):
    test_path = os.path.join(data_path, f"post_{i:03d}.png")
    if os.path.exists(test_path):
        test_images.append(test_path)

if len(test_images) == 0:
    print("No test images found!")
    sys.exit(1)

print(f"Found {len(test_images)} test images")

# Test first 3 images
for idx, test_path in enumerate(test_images[:3]):
    print(f"\n  Testing image {idx+1}: {os.path.basename(test_path)}")
    try:
        mask, confidence = predictor.predict(test_path)
        stats = predictor.get_stats(mask)
        print(f"  Flood coverage: {stats['percentage']:.1f}% ({stats['severity']})")
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        img = plt.imread(test_path)
        axes[0].imshow(img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(mask[:,:,0], cmap='Blues', vmin=0, vmax=1)
        axes[1].set_title('Flood Mask')
        axes[1].axis('off')
        
        overlay = predictor.overlay(test_path, mask)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"  Error testing image: {e}")

print("\n" + "="*60)
print("TESTING COMPLETE!")
print("="*60)
print("\nNext: Run streamlit run 04_app.py")