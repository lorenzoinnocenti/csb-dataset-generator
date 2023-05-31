import cv2
import numpy as np
from numpy import random

from package.psf_generation.csb_psf import CSBPSF
from package.psf_generation.center_kernel import center_kernel


class ConstantTrajectory:

    def process_image(self, image):
        # gets as input image, in the format of an ndarray
        exposure = random.choice(self.exposures)
        kernel = self.psf.generate(exposure=exposure)
        kernel = center_kernel(kernel)
        blurred_image = image / 255
        lambda_poisson = self.lambda_poisson/255
        sigma_gauss = self.sigma_gauss/255
        # blur image
        blurred_image = cv2.filter2D(src=blurred_image, ddepth=-1, kernel=kernel)
        blurred_image = blurred_image.clip(min=0, max=1)
        # add noise
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

    def __init__(self, 
                 trajectory,
                 psf_size,
                 lambda_poisson=48000,
                 sigma_gauss=4/255,
                 exposures=(1, 1/2, 1/4, 1/8, 1/16),
                 return_kernel=False
                 ):
        self.trajectory = trajectory
        self.sigma_gauss = sigma_gauss
        self.lambda_poisson = lambda_poisson
        self.exposures = exposures
        self.return_kernel = return_kernel
        self.psf = CSBPSF(trajectory,
                          psf_size=psf_size,
                          )
