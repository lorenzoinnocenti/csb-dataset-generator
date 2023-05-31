from skimage.draw import disk
import numpy as np
from math import ceil


def disk_psf_generation(psf_size, radius_size, precision=4):
    centr = (psf_size) / 2
    coordinates = disk((centr * precision, centr * precision), radius_size * precision)
    coordinates = np.transpose(coordinates)
    kernel = np.zeros(np.array([psf_size, psf_size]))
    for r, c in coordinates:
        r_int = int(ceil(r / precision - 0.5))
        c_int = int(ceil(c / precision - 0.5))
        kernel[r_int, c_int] += 1
    kernel = kernel / precision
    sum = np.sum(kernel)
    return kernel / sum


if __name__ == "__main__":

    psf_size = 128
    disk_radius = 8
    precision = 4

    from PIL import Image
    import timeit
    start = timeit.default_timer()

    kernel = disk_psf_generation(psf_size, disk_radius, precision=precision)
    from matplotlib import pyplot as plt

    plt.imshow(kernel, cmap="gray")

    kernel = kernel/np.max(kernel)
    kernel_img = Image.fromarray((kernel * 255).astype(np.uint8))
    kernel_img.save("kernel3.png")

    stop = timeit.default_timer()
    print('Time: ', stop - start)
    plt.show()
