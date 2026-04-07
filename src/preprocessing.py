import cv2
import os
import numpy as np
from skimage.filters import threshold_sauvola
import config

class ModiPreprocessor:
    def __init__(self):
        self.size = config.IMAGE_SIZE

    def apply_otsu(self, img):
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def apply_sauvola(self, img):
        thresh_val = threshold_sauvola(img, window_size=25)
        binary_sauvola = img > thresh_val
        return (binary_sauvola * 255).astype(np.uint8)

    def apply_tozero(self, img):
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
        return thresh

    def process_and_save(self, method):
        print(f"Starting {method} preprocessing...")
        
        # Mapping for your new folder structure
        for split in ['train', 'test']:
            split_path = os.path.join(config.RAW_DATA_DIR, split)
            if not os.path.exists(split_path): continue

            for label in os.listdir(split_path):
                label_path = os.path.join(split_path, label)
                if not os.path.isdir(label_path): continue

                target_dir = os.path.join(config.PROCESSED_DIR, method, split, label)
                os.makedirs(target_dir, exist_ok=True)

                for img_name in os.listdir(label_path):
                    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                    
                    img_path = os.path.join(label_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None: continue

                    # Resize as per model requirements (Upscaling 32x32 to 75x75)
                    img = cv2.resize(img, self.size)

                    if method == 'otsu': processed = self.apply_otsu(img)
                    elif method == 'sauvola': processed = self.apply_sauvola(img)
                    elif method == 'tozero': processed = self.apply_tozero(img)
                    else: processed = img

                    cv2.imwrite(os.path.join(target_dir, img_name), processed)
        print(f"Preprocessing for {method} completed.")

def run_preprocessing(method):
    processor = ModiPreprocessor()
    processor.process_and_save(method)