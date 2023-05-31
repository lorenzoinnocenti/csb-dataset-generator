from PIL import Image
import cv2
from PyConvBlur.psf_generation.motion_psf import *
import numpy as np
from numpy import random
import matplotlib.pyplot as plt


def blur_raw(y, psf_list, lambda_poisson=1, sigma_gauss=0, seed=None):
    if seed is not None:
        random.seed(seed)

    blurred_images = []
    noisy_blurred_images = []

    # normalize all images
    y = y / 255

    # rescale the original image
    y = y * lambda_poisson

    # generate distorted image
    for i, p in enumerate(psf_list):
        blurred_images.append(cv2.filter2D(src=y, ddepth=-1, kernel=p))

    # add noise
    gauss_noise = random.normal(loc=0.0, scale=sigma_gauss, size=y.shape)

    for i, img in enumerate(blurred_images):
        # poiss_noise_img = random.poisson(img.clip(0))
        poiss_noise_img = random.poisson(img)
        noisy_blurred_images.append(gauss_noise + poiss_noise_img)

    fig = plt.figure(figsize=(16, 8))

    psf_max = np.max(np.max(psf_list))

    for i, p in enumerate(psf_list):
        fig.add_subplot(2, len(psf_list), i + 1)
        normalized_psf = Image.fromarray(p * 255 / psf_max)
        plt.imshow(normalized_psf)
        fig.add_subplot(2, len(psf_list), i + 1 + len(psf_list))
        plt.imshow(noisy_blurred_images[i] / np.max(blurred_images[i]))

    plt.show()


if __name__ == "__main__":
    from PyConvBlur.psf_generation.trajectory_generator import TrajectoryGenerator
    import matplotlib.pyplot as plt
    from PIL import Image
    psf_size = 128
    num_t = psf_size*1000
    for i in range(0, 10):
        print(i)
        trajectory = TrajectoryGenerator(traj_size=psf_size, num_t=num_t)
        # plt.imshow(kernel, cmap='gray')
        # plt.show()
        psf = PSF(trajectory,
                  psf_size=psf_size)
        kernel = psf.generate()
        plt.imshow(kernel, cmap='gray')
        plt.show()
        Image.fromarray((kernel * 255).astype(np.uint8)).save('kernels/' + str(i) + '.png')


