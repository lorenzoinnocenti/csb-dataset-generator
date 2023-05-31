import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import random

from package.random_degradation.constant_trajectory import ConstantTrajectory

trajpath: str = 'PyConvBlur/psf_generation/trajectories/'
trajname: str = 'traj2'
dataset_path = '../../../../dataset/imagenet/ILSVRC/Data/CLS-LOC/test'
output_folder = 'PyConvBlur/traj2_test/exp16'

nr_images = 100

psf_size = 32
motion_size = 16
shuffle = False

# take the first n images from imagenet test, degrade them, ad try to restore them with wiener filtering

if __name__ == "__main__":
    with open(trajpath+trajname+'.npy', 'rb') as f:
        trajectory = np.load(f)
    degradation = ConstantTrajectory(trajectory, psf_size=psf_size, return_kernel=True, exposures=(16 / 16,))
    # degradation = RandomMotionBlur(psf_size=psf_size, return_kernel=True, max_total_length=motion_size, min_total_length=1)
    
    image_paths = []
    image_names = []
    for name in os.listdir(dataset_path):
        path = os.path.join(dataset_path, name)
        if os.path.isdir(path):
            for image_name in os.listdir(path):
                image_paths.append(os.path.join(path, image_name))
                image_names.append(image_name)
        else:
            image_paths.append(os.path.join(dataset_path, name))
            image_names.append(name)
            
    image_paths = sorted(image_paths)
    image_names = sorted(image_names)
    
    if shuffle: 
        random.seed(10)
        temp = list(zip(image_paths, image_names))
        random.shuffle(temp)
        image_paths, image_names = zip(*temp)
    
    for i, image_path in enumerate(image_paths[:nr_images]):
        print(image_names[i])
        # load image
        sharp_img = Image.open(image_path)
        plt.imshow(sharp_img)
        plt.show()
        sharp_img = np.array(sharp_img)
        # degrade image
        blurred_image, kernel = degradation.process_image(sharp_img)
        # save image
        blurred_image = Image.fromarray((blurred_image).astype(np.uint8))
        plt.imshow(blurred_image)
        plt.show()
        # kernel_image = Image.fromarray((kernel*255).astype(np.uint8))
        kernel_image = kernel*255
        blurred_image.save(os.path.join(output_folder, image_names[i]))
        # kernel_image.save(os.path.join(kernel_folder, image_names[i]))
        # cv2.imwrite(os.path.join(kernel_folder, image_names[i]), kernel_image)
