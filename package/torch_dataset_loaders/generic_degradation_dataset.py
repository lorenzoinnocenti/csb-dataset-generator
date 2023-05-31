import numpy as np
from PIL import Image

from .single_path_image_dataset import SinglePathImageDataset


class GenericDegradationDataset(SinglePathImageDataset):
    """
    Class that handles the data loading of images degraded with a processing function which has images as both input
    and output. It can be used with the process_image function from the random_degradation classes.
    The processing function is defined during initialization.
    """

    def __init__(self, DATASET_PATH, image_size, degrade=None, crop='random'):
        super().__init__(DATASET_PATH, image_size, crop)
        self.degrade = degrade

    def __getitem__(self, idx):
        sharp = Image.open(self.image_paths[idx])
        blurred = self.degrade(np.float32(sharp))  # kernel returned is discarded
        blurred = Image.fromarray(np.uint8(blurred))
        blurred, sharp = self.transform(blurred, sharp)
        # print("motion")
        return blurred, sharp
