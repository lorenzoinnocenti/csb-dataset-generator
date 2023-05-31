import os
import random

from PIL import Image
from ..dataset_loader.image_dataset import ImageDataset


class RealTestDataset(ImageDataset):
    # loader for the test set for the imagenet synthetic blur dataset

    def __init__(self, sharp_path, blur_path, image_size):
        super().__init__(image_size, crop='center', hflip=False, rot_degrees=0, 
                         brightness_jitter=0, saturation_jitter=0, contrast_jitter=0)
        self.sharp_paths = []
        self.sharp_names = []
        self.blur_paths = []
        self.blur_names = []
        for name in os.listdir(sharp_path):
            path = os.path.join(sharp_path, name)
            if os.path.isdir(path):
                for image_name in os.listdir(path):
                    self.sharp_paths.append(os.path.join(path, image_name))
                    self.sharp_names.append(image_name)
            else:
                self.sharp_paths.append(os.path.join(sharp_path, name))
                self.sharp_names.append(name)
        for name in os.listdir(blur_path):
            path = os.path.join(blur_path, name)
            if os.path.isdir(path):
                for image_name in os.listdir(path):
                    self.blur_paths.append(os.path.join(path, image_name))
                    self.blur_names.append(image_name)
            else:
                self.blur_paths.append(os.path.join(blur_path, name))
                self.blur_names.append(name)

    def __getitem__(self, idx):
        blur_img = Image.open(self.blur_paths[idx])
        sharp_idx = self.sharp_names.index(self.blur_names[idx])
        sharp_img = Image.open(self.sharp_paths[sharp_idx])
        return self.transform(blur_img, sharp_img)

    def __len__(self):
        return len(self.blur_paths)
