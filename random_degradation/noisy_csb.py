import cv2
import numpy as np
from numpy import random

from psf_generation.csb_psf import CSBPSF
from psf_generation.trajectory_generator import TrajectoryGenerator
from psf_generation.center_kernel import center_kernel


class NosyCSB:

    def process_image(self, image):
        # gets as input image, in the format of an ndarray
        self.seed = self.seed + 1
        exposure = random.choice(self.exposures)
        lambda_poisson = random.choice(self.lambdas)/255
        sigma_gauss = random.choice(self.sigmas)/255
        trajectory: np.ndarray = TrajectoryGenerator(
            traj_size=self.psf_size,
            anxiety=self.anxiety,
            centripetal_term=self.centripetal_term,
            gaussian_term=self.gaussian_term,
            freq_big_shakes=self.freq_big_shakes,
            num_t=self.num_t,
            max_total_length=self.max_total_length,
            seed=self.seed
            ).generate()
        psf = CSBPSF(trajectory,
                     psf_size=self.psf_size,
                     )
        kernel = psf.generate(exposure=exposure)
        kernel = center_kernel(kernel)
        blurred_image = image / 255
        # blur image
        blurred_image = cv2.filter2D(src=blurred_image, ddepth=-1, kernel=kernel)
        blurred_image = blurred_image.clip(min=0, max=1)
        # add noise
        if self.apply_noise: 
            gauss_noise = random.normal(loc=0.0, scale=sigma_gauss, size=blurred_image.shape)
            blurred_image = random.poisson(blurred_image * lambda_poisson, size=blurred_image.shape) / lambda_poisson
            blurred_image = gauss_noise + blurred_image
        blurred_image = blurred_image / exposure
        # import matplotlib.pyplot as plt
        # plt.imshow(image/255)
        # plt.show()
        # plt.imshow(blurred_image)
        # plt.show()
        # plt.imshow(kernel, cmap="gray")
        # plt.show()
        # bring back the range in 0-1
        blurred_image = blurred_image.clip(min=0, max=1)
        blurred_image = blurred_image * 255
        if self.return_kernel:
            return blurred_image, kernel
        else: 
            return blurred_image


    def __init__(self, psf_size=64, anxiety=None, centripetal_term=None,
                 gaussian_term=None, freq_big_shakes=None,
                 num_t=2000, max_total_length=None, seed=None,
                 lambdas=(48000, 192000, 768000, 3072000, 12288000),
                 sigmas=(0, 1, 2, 4,),
                 exposures=(8/8, 7/8, 6/8, 5/8, 4/8, 3/8, 2/8,),
                 return_kernel=False,
                 apply_noise=True,
                 ):
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
        self.sigmas = sigmas
        self.lambdas = lambdas
        self.exposures = exposures
        self.return_kernel = return_kernel
        self.apply_noise = apply_noise
        