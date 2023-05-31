import numpy as np

class CSBPSF:
    """
    Class that handles the sampling of the trajectory in a 2d kernel.
    It takes as input for the constructor the complex valued ndarray that represent the trajectory and the size of the
    kernels to be generated. When the generate function is called, the trajectory is sampled proportionally to the
    exposure value, and output as a 2d real valued ndarray.
    """
    def __init__(self, trajectory: np.ndarray, psf_size=64):
        self.trajectory = trajectory
        self.psf_size = np.array([psf_size, psf_size])

    def generate(self, exposure=1):
        num_t = self.trajectory.size

        # generate the PSFs
        kernel = np.zeros(self.psf_size)

        # sample the trajectory until time T
        for t in range(0, num_t):
            if (exposure * num_t >= t + 1):
                t_proportion = 1
            elif (exposure * num_t >= t):
                t_proportion = (exposure * num_t) - t
            else:
                t_proportion = 0
            # t_proportion = int(t_proportion)

            m2 = int(min(self.psf_size[1] - 1, max(1, np.floor(np.real(self.trajectory[t])))))
            M2 = m2 + 1
            m1 = int(min(self.psf_size[0] - 1, max(1, np.floor(np.imag(self.trajectory[t])))))
            M1 = m1 + 1

            # linear interpolation
            kernel[m1, m2] = kernel[m1, m2] + t_proportion * triangle_fun_prod(np.real(self.trajectory[t]) - m2, np.imag(self.trajectory[t]) - m1)
            kernel[m1, M2] = kernel[m1, M2] + t_proportion * triangle_fun_prod(np.real(self.trajectory[t]) - M2, np.imag(self.trajectory[t]) - m1)
            kernel[M1, m2] = kernel[M1, m2] + t_proportion * triangle_fun_prod(np.real(self.trajectory[t]) - m2, np.imag(self.trajectory[t]) - M1)
            kernel[M1, M2] = kernel[M1, M2] + t_proportion * triangle_fun_prod(np.real(self.trajectory[t]) - M2, np.imag(self.trajectory[t]) - M1)

        kernel = kernel/num_t
        self.kernel = kernel
        return kernel

def triangle_fun_prod(d1, d2):
    x1 = max(0, (1 - np.abs(d1)))
    x2 = max(0, (1 - np.abs(d2)))
    return x1 * x2


if __name__ == "__main__":
    from psf_generation.trajectory_generator import TrajectoryGenerator
    import matplotlib.pyplot as plt
    from PIL import Image
    psf_size = 128
    for i in range(0, 100):
        print(i)
        trajectory: np.ndarray = TrajectoryGenerator(traj_size=psf_size).generate()
        psf = CSBPSF(trajectory,
                     psf_size=psf_size)
        kernel = psf.generate()
        plt.imshow(kernel, cmap='gray')
        plt.show()
        kernel = kernel/np.max(kernel)
        Image.fromarray((kernel * 255).astype(np.uint8)).save('kernels/' + str(i) + '.png')


