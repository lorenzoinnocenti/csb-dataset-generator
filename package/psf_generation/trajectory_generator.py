import numpy as np
from numpy import random


class TrajectoryGenerator:

    def __init__(self, traj_size=64, anxiety=None, centripetal_term=None, gaussian_term=None, freq_big_shakes=None,
                 num_t=2000, max_total_length=None, seed=None):
        if (seed != None):
            random.seed(seed)

        # to make sure to stay within the borders of the canvas
        if max_total_length is None:
            self.max_total_length = int(traj_size/2)
        else:
            self.max_total_length = max_total_length

        if (anxiety == None):
            self.anxiety = 0.01 * random.random()
            # FIXME the original coefficient of 0.1 generates VERY shakey kernels
        else:
            self.anxiety = anxiety

        # term determining, at each sample, the strengh of the component leating towards the previous position
        if centripetal_term == None:
            self.centripetal_term = 0.7 * random.random()
        else:
            self.centripetal_term = centripetal_term

        # term determining, at each sample, the random component of the new direction
        if gaussian_term == None:
            self.gaussian_term = 10 * random.random()
        else:
            self.gaussian_term = gaussian_term

        # probability of having a big shake, e.g. due to pressing camera button or abrupt hand movements
        if freq_big_shakes == None:
            self.freq_big_shakes = 0.2 * random.random()
        else:
            self.freq_big_shakes = freq_big_shakes

        self.num_t = num_t
        self.traj_size = traj_size

    def generate(self):
        # Generate x(t), Discrete Random Motion Trajectory  in Continuous Domain

        # v is the initial velocity vector, initialized at random direction
        init_angle = 2 * np.pi * random.random()

        # initial velocity vector having norm 1
        v0 = np.complex(np.cos(init_angle), np.sin(init_angle))
        # the speed of the initial velocity vector
        v = v0 * self.max_total_length / (self.num_t - 1)

        if self.anxiety > 0:
            v = v0 * self.anxiety  

        x = np.zeros(self.num_t).astype(complex)

        for t in range(0, self.num_t - 1):  # TODO in matlab goes from 1 to num_t-1
            # determine if there is an abrupt (impulsive) shake
            if random.random() < self.freq_big_shakes * self.anxiety:
                # the new direction is the opposite of the previous one, plus or minus half radiant
                next_direction = 2 * v * (np.exp(1j * (np.pi + (random.random() - 0.5))))
            else:
                next_direction = 0

            # determine the random component motion vector at the next step
            # the difference of v is computed as next direction (whitch is 0 if there are no big shakes)
            dv = next_direction + \
                 self.anxiety * (  # all multiplied by anxiety factor
                         self.gaussian_term * (random.normal(loc=0, scale=1) + 1j * random.normal(loc=0,
                                                                                                  scale=1))  # random gaussian component, in both x and y
                         - self.centripetal_term * x[t]
                     # centripetal component, computed as the centripetal term multiplied by the direction of the previous point FIXME: shouldn't it be x[0]?
                 ) * (self.max_total_length / (self.num_t - 1))  # normalization
            v = v + dv

            # velocity vector normalization
            v = (v / abs(v)) * self.max_total_length / (self.num_t - 1)

            # update particle position
            x[t + 1] = x[t] + v

        # Compute distance from center
        center = (self.traj_size)/2
        dist_x = np.average(np.real(x)) - center
        dist_y = np.average(np.imag(x)) - center

        # Centering the trajectory
        x = x - dist_x
        x = x - 1j * dist_y

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # fig.add_subplot(1, 1, 1)
        # plt.scatter(np.real(x), np.imag(x), s=.01)
        # plt.xlim(0, self.traj_size)
        # plt.ylim(0, self.traj_size)
        # plt.show()

        return x
