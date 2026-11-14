import os
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
