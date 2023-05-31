import numpy as np
import cv2
from PIL import Image

from .single_path_image_dataset import SinglePathImageDataset


class ConstantKernelDataset(SinglePathImageDataset):

    def __init__(self, DATASET_PATH, psf_kernel, image_size, crop='random'):
        super().__init__(DATASET_PATH, image_size, crop)
        sum = np.sum(psf_kernel)
        self.psf_kernel = psf_kernel/sum    
    
    def __getitem__(self, idx):
        sharp = Image.open(self.image_paths[idx])
        blurred = cv2.filter2D(src=np.float32(sharp), ddepth=-1, kernel=self.psf_kernel)
        blurred = Image.fromarray(np.uint8(blurred))
        blurred, sharp = self.transform(blurred, sharp)
        return blurred, sharp