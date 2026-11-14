import tensorflow as tf
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
