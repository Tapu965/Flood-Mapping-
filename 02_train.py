"""
TRAINING SCRIPT - RUN AFTER 01_complete_setup.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.data_loader import DataLoader
from models.unet_model import FloodUNet

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Force CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("="*60)
print("STEP 1: TRAINING FLOOD DETECTION MODEL")
print("="*60)
print(f"Working directory: {BASE_DIR}")

# Check if data exists
data_path = os.path.join(BASE_DIR, 'data', 'raw')
if not os.path.exists(data_path):
    print(f"\nERROR: Data directory not found at {data_path}")
    print("Please run 01_complete_setup.py first")
    sys.exit(1)

# Load data
print("\nLoading data...")
loader = DataLoader('config.yaml')
X, y = loader.load_data()

if len(X) == 0:
    print("\nERROR: No data loaded! Please run 01_complete_setup.py first")
    sys.exit(1)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.split_data(X, y)

print(f"\nDataset sizes:")
print(f"  Train: {len(X_train)} images")
print(f"  Validation: {len(X_val)} images")
print(f"  Test: {len(X_test)} images")

# Build model
print("\nBuilding U-Net model...")
model_builder = FloodUNet('config.yaml')
model = model_builder.build_model()  # First build the model
model = model_builder.compile_model()  # Then compile it
model.summary()

# Train
print("\nTraining model (10-15 minutes)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=4,
    epochs=15,
    verbose=1
)

# Save model
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'flood_model.h5')
model.save(model_path)
print(f"\nModel saved to: {model_path}")

# Evaluate
print("\nEvaluating...")
loss, acc, iou = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Loss: {loss:.4f}")
print(f"  Test Accuracy: {acc:.4f}")
print(f"  Test IoU: {iou:.4f}")

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'training_history.png'))
plt.show()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nNext: Run python 03_test.py")