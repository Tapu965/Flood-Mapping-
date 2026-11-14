"""
COMPLETE SETUP SCRIPT - RUN THIS FIRST
This script creates all directories, files, and synthetic dataset
"""

import os
import numpy as np
from PIL import Image

# Get the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("="*60)
print("FLOOD MAPPING PROJECT - COMPLETE SETUP")
print("="*60)
print(f"Working directory: {BASE_DIR}")

# Create directory structure with absolute paths
print("\nCreating project structure...")
dirs = [
    os.path.join(BASE_DIR, 'preprocessing'),
    os.path.join(BASE_DIR, 'models'),
    os.path.join(BASE_DIR, 'inference'),
    os.path.join(BASE_DIR, 'data', 'raw'),
    os.path.join(BASE_DIR, 'data', 'processed'),
    os.path.join(BASE_DIR, 'data', 'samples'),
    os.path.join(BASE_DIR, 'models', 'trained_models'),
    os.path.join(BASE_DIR, 'logs')
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"  Created: {d}")

# Create synthetic dataset
print("\nCreating synthetic dataset...")
raw_data_dir = os.path.join(BASE_DIR, 'data', 'raw')

for i in range(30):
    # Create base image
    img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
    
    # Create random flood mask (circular blobs)
    mask = np.zeros((128, 128), dtype=np.uint8)
    num_floods = np.random.randint(1, 4)
    
    for _ in range(num_floods):
        x = np.random.randint(30, 100)
        y = np.random.randint(30, 100)
        r = np.random.randint(15, 30)
        yy, xx = np.ogrid[:128, :128]
        circle = (xx - x)**2 + (yy - y)**2 <= r**2
        mask[circle] = 255
    
    # Create pre and post images
    pre_img = img.copy()
    post_img = img.copy()
    post_img[mask > 0] = [30, 80, 200]  # Blue for water
    
    # Save files with full paths
    Image.fromarray(pre_img).save(os.path.join(raw_data_dir, f"pre_{i:03d}.png"))
    Image.fromarray(post_img).save(os.path.join(raw_data_dir, f"post_{i:03d}.png"))
    Image.fromarray(mask).save(os.path.join(raw_data_dir, f"mask_{i:03d}.png"))
    
    if (i+1) % 10 == 0:
        print(f"  Created {i+1} images")

print(f"  Created 30 synthetic images in: {raw_data_dir}")

# Create config.yaml with absolute paths
print("\nCreating configuration files...")

config_path = os.path.join(BASE_DIR, 'config.yaml')
with open(config_path, 'w') as f:
    f.write(f"""# CPU-OPTIMIZED CONFIGURATION
data:
  image_size: [128, 128]
  num_channels: 3
  max_samples: 100
  base_dir: "{BASE_DIR.replace('\\', '\\\\')}"

model:
  input_shape: [128, 128, 6]
  num_classes: 1
  batch_size: 4
  epochs: 15
  learning_rate: 0.0005

paths:
  model_save_path: "{os.path.join(BASE_DIR, 'models', 'trained_models', 'flood_model.h5').replace('\\', '\\\\')}"
  data_path: "{raw_data_dir.replace('\\', '\\\\')}"
""")

# Create preprocessing/__init__.py
init_path = os.path.join(BASE_DIR, 'preprocessing', '__init__.py')
with open(init_path, 'w') as f:
    f.write("""from .data_loader import DataLoader
from .image_preprocessor import ImagePreprocessor
__all__ = ['DataLoader', 'ImagePreprocessor']
""")

# Create preprocessing/data_loader.py with absolute paths
loader_path = os.path.join(BASE_DIR, 'preprocessing', 'data_loader.py')
with open(loader_path, 'w') as f:
    f.write("""import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import yaml

class DataLoader:
    def __init__(self, config_path='config.yaml'):
        # Get the directory of this script
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(self.script_dir)
        
        # Load config
        config_full_path = os.path.join(self.base_dir, config_path)
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.image_size = tuple(self.config['data']['image_size'])
        self.data_path = self.config['paths']['data_path']
        
    def load_data(self):
        print("Loading dataset...")
        print(f"Looking for data in: {self.data_path}")
        
        # Check if directory exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        pre_images, post_images, masks = [], [], []
        
        # Get all files
        try:
            files = os.listdir(self.data_path)
        except Exception as e:
            print(f"Error reading directory: {e}")
            return np.array([]), np.array([])
        
        pre_files = sorted([f for f in files if f.startswith('pre_')])[:100]
        
        if len(pre_files) == 0:
            print("No pre_ files found! Please run 01_complete_setup.py first")
            return np.array([]), np.array([])
        
        print(f"Found {len(pre_files)} images")
        
        for pre_file in pre_files:
            base = pre_file.replace('pre_', '')
            post_file = f"post_{base}"
            mask_file = f"mask_{base}"
            
            pre_path = os.path.join(self.data_path, pre_file)
            post_path = os.path.join(self.data_path, post_file)
            mask_path = os.path.join(self.data_path, mask_file)
            
            # Check if files exist
            if not all(os.path.exists(p) for p in [pre_path, post_path, mask_path]):
                print(f"Warning: Missing files for {base}")
                continue
            
            # Load and process images
            pre = Image.open(pre_path).resize(self.image_size)
            post = Image.open(post_path).resize(self.image_size)
            mask = Image.open(mask_path).resize(self.image_size)
            
            pre_images.append(np.array(pre)/255.0)
            post_images.append(np.array(post)/255.0)
            masks.append(np.expand_dims(np.array(mask)/255.0, axis=-1))
        
        if len(pre_images) == 0:
            print("No images could be loaded!")
            return np.array([]), np.array([])
        
        X = np.concatenate([np.array(pre_images), np.array(post_images)], axis=-1)
        y = np.array(masks)
        print(f"Loaded {len(X)} images successfully")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        return X, y
    
    def split_data(self, X, y):
        if len(X) == 0:
            print("No data to split!")
            return (np.array([]), np.array([])), (np.array([]), np.array([])), (np.array([]), np.array([]))
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
""")

# Create preprocessing/image_preprocessor.py
preproc_path = os.path.join(BASE_DIR, 'preprocessing', 'image_preprocessor.py')
with open(preproc_path, 'w') as f:
    f.write("""import numpy as np
import cv2
import albumentations as A

class ImagePreprocessor:
    def __init__(self):
        self.image_size = (128, 128)
        self.augment = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ])
    
    def preprocess(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        return image
""")

# Create models/__init__.py
models_init = os.path.join(BASE_DIR, 'models', '__init__.py')
with open(models_init, 'w') as f:
    f.write("""from .unet_model import FloodUNet
__all__ = ['FloodUNet']
""")

# Create models/unet_model.py
unet_path = os.path.join(BASE_DIR, 'models', 'unet_model.py')
with open(unet_path, 'w') as f:
    f.write("""import tensorflow as tf
from tensorflow.keras import layers, models
import yaml
import os

class FloodUNet:
    def __init__(self, config_path='config.yaml'):
        # Get base directory
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(self.script_dir)
        
        # Load config
        config_full_path = os.path.join(self.base_dir, config_path)
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.input_shape = tuple(self.config['model']['input_shape'])
        self.num_classes = self.config['model']['num_classes']
        
        # CPU optimizations
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)
    
    def build_model(self):
        inputs = layers.Input(self.input_shape)
        
        # Encoder
        c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D()(c1)
        
        c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D()(c2)
        
        # Bridge
        c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
        
        # Decoder
        u1 = layers.UpSampling2D()(c3)
        u1 = layers.concatenate([u1, c2])
        c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
        c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)
        
        u2 = layers.UpSampling2D()(c4)
        u2 = layers.concatenate([u2, c1])
        c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
        c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(c5)
        
        # Output
        outputs = layers.Conv2D(self.num_classes, 1, activation='sigmoid')(c5)
        
        self.model = models.Model(inputs, outputs)
        return self.model
    
    def compile_model(self):
        if self.model is None:
            self.build_model()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config['model']['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.BinaryIoU()]
        )
        return self.model
""")

# Create inference/__init__.py
infer_init = os.path.join(BASE_DIR, 'inference', '__init__.py')
with open(infer_init, 'w') as f:
    f.write("""from .predictor import FloodPredictor
__all__ = ['FloodPredictor']
""")

# Create inference/predictor.py
predictor_path = os.path.join(BASE_DIR, 'inference', 'predictor.py')
with open(predictor_path, 'w') as f:
    f.write("""import numpy as np
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
        
        if model_path:
            full_model_path = os.path.join(self.base_dir, model_path)
            self.model = tf.keras.models.load_model(full_model_path, compile=False)
    
    def predict(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        img = Image.open(image_path).resize(self.image_size)
        img_array = np.array(img)/255.0
        X = np.concatenate([img_array, img_array], axis=-1)
        X = np.expand_dims(X, 0)
        pred = self.model.predict(X, verbose=0)[0]
        return (pred > 0.5).astype(np.uint8), pred
    
    def overlay(self, image_path, mask):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        mask = cv2.resize(mask.astype(np.float32), (256, 256))
        overlay = img.copy()
        overlay[mask > 0.5] = [255, 0, 0]
        return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    def get_stats(self, mask):
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        pct = (np.sum(mask > 0.5) / mask.size) * 100
        if pct < 10:
            severity = "Minor"
        elif pct < 30:
            severity = "Moderate"
        elif pct < 50:
            severity = "Severe"
        else:
            severity = "Catastrophic"
        return {'percentage': float(pct), 'severity': severity}
""")

print("\n" + "="*60)
print("SETUP COMPLETE!")
print("="*60)
print(f"\nAll files created in: {BASE_DIR}")
print("\nNext steps:")
print("1. Open Command Prompt as Administrator")
print("2. Navigate to your project folder:")
print(f"   cd {BASE_DIR}")
print("3. Install requirements:")
print("   pip install -r requirements.txt")
print("4. Run training:")
print("   python 02_train.py")