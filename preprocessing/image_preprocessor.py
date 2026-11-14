import numpy as np
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
