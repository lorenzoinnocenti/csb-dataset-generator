import cv2
import numpy as np
from numpy import random

from psf_generation.disk_psf import disk_psf_generation


class DiskBlur:
    """
    Class used to generate blurred images, by convolution with kernel generated with the disk_psf_generation function.
    The class is initialized with a range of radii. Each time an image is passed to the
    process_image function, a new kernel is generated within the specified range, and it's used to degrade the image.
    """
    def process_image(self, image):
        # radius_size = random.randint(self.min_disk_radius, self.max_disk_radius)
        radius_size = random.uniform(self.min_disk_radius, self.max_disk_radius)
        kernel = disk_psf_generation(self.psf_size, radius_size, precision=10)
        # import matplotlib.pyplot as plt
        # plt.imshow(kernel, cmap="gray")
        # plt.show()
        sum = np.sum(kernel)
        kernel = kernel / sum
        corrupted_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
        return corrupted_image

    def __init__(self, psf_size=64, max_disk_radius=32, min_disk_radius=0):
        self.psf_size = psf_size
        self.max_disk_radius = max_disk_radius
        self.min_disk_radius = min_disk_radius