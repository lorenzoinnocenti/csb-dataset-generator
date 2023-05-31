import numpy as np
import matplotlib.pyplot as plt

from .csb_psf import CSBPSF
from .trajectory_generator import TrajectoryGenerator
from .center_kernel import center_kernel

psf_size = 64
traj_length = 32
trajpath: str = 'PyConvBlur/psf_generation/trajectories/'
trajname: str = 'traj1'

if __name__ == "__main__":
    trajectory: np.ndarray = TrajectoryGenerator(
            traj_size=psf_size,
            max_total_length=traj_length,
            ).generate()
    psf = CSBPSF(trajectory,
                 psf_size=psf_size,
                 )
    kernel = psf.generate()
    kernel = center_kernel(kernel)
    sum = np.sum(kernel)
    kernel = kernel / sum
    plt.imshow(kernel, cmap="gray")
    plt.show()
    with open(trajpath+trajname+'.npy', 'wb') as f:
        np.save(f, trajectory)
    # with open(trajpath+trajname+'.npy', 'rb') as f:
    #     trajectory = np.load(f)
    # psf = MotionPSF(trajectory,
    #                 psf_size=psf_size,
    #                 )
    # kernel = psf.generate()
    # kernel = center_kernel(kernel)
    # sum = np.sum(kernel)
    # kernel = kernel / sum
    # plt.imshow(kernel, cmap="gray")
    # plt.show()