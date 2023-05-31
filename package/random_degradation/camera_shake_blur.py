import cv2
import numpy as np
from numpy import random

from package.psf_generation.csb_psf import CSBPSF
from package.psf_generation.trajectory_generator import TrajectoryGenerator
from package.psf_generation.center_kernel import center_kernel


class CameraShakeBlur:

    def process_image(self, image):
        # gets as input image, in the format of an ndarray,
        self.seed = self.seed + 1
        traj_length = random.uniform(self.min_total_length, self.max_total_length)
        trajectory: np.ndarray = TrajectoryGenerator(
            traj_size=self.psf_size,
            anxiety=self.anxiety,
            centripetal_term=self.centripetal_term,
            gaussian_term=self.gaussian_term,
            freq_big_shakes=self.freq_big_shakes,
            num_t=self.num_t,
            max_total_length=traj_length,
            seed=self.seed
            ).generate()
        psf = CSBPSF(trajectory,
                     psf_size=self.psf_size,
                     )
        kernel = psf.generate(exposure=self.exposure)
        kernel = center_kernel(kernel)
        sum = np.sum(kernel)
        kernel = kernel / sum
        # import matplotlib.pyplot as plt
        # plt.imshow(kernel, cmap="gray")
        # plt.show()
        corrupted_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
        if self.return_kernel:
            return corrupted_image, kernel
        else: 
            return corrupted_image

    def __init__(self, psf_size=64, anxiety=None, centripetal_term=None, gaussian_term=None, freq_big_shakes=None,
                 num_t=2000, max_total_length=None, min_total_length=1, seed=None, exposure=1, return_kernel=False):
        if seed is not None:
            self.seed = seed
        else:
            self.seed = 123
        self.num_t = num_t
        self.anxiety = anxiety
        self.centripetal_term = centripetal_term
        self.gaussian_term = gaussian_term
        self.freq_big_shakes = freq_big_shakes
        self.psf_size = psf_size
        self.max_total_length = max_total_length
        self.min_total_length = min_total_length
        self.exposure = exposure
        self.return_kernel = return_kernel