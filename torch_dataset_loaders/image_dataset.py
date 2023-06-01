import torch
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
from ..utils.auto_resizer import resize_if_needed
from PIL import Image, ImageEnhance


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_size=None, crop='random', hflip=False, rot_degrees=0,
                 brightness_jitter=0, saturation_jitter=0, contrast_jitter=0, resize_factor=1):
        if crop != 'center' and crop != 'random' and crop != 'none':
            raise Exception('Crop not valid')
        self.crop = crop
        self.hflip = hflip
        self.rot_degrees = rot_degrees
        self.image_size = image_size
        self.brightness_jitter = brightness_jitter
        self.saturation_jitter = saturation_jitter
        self.contrast_jitter = contrast_jitter
        self.resize_factor = resize_factor

    def transform(self, image, label):
        if image.mode != "RGB":
            image = image.convert("RGB")
        if label.mode != "RGB":
            label = label.convert("RGB")
            
        # resize
        if self.resize_factor!=1:
            new_size_x = int(image.size[1]*self.resize_factor)
            new_size_y = int(image.size[0]*self.resize_factor)
            resize_transform = transforms.Resize((new_size_x, new_size_y))
            image = resize_transform(image)
            label = resize_transform(label)
        
        image = resize_if_needed(image, self.image_size)
        label = resize_if_needed(label, self.image_size)

        # augmentations
        # horiz flip
        if random.random() > 0.5 and self.hflip:
            image = TF.hflip(image)
            label = TF.hflip(label)
        # random rotation
        if self.rot_degrees > 0:
            random_rot_degrees = random.random()*(2*self.rot_degrees)-self.rot_degrees
            image = transforms.functional.rotate(image, random_rot_degrees, interpolation=TF.InterpolationMode.BILINEAR)
            label = transforms.functional.rotate(label, random_rot_degrees, interpolation=TF.InterpolationMode.BILINEAR)
        # color jitter
        if self.brightness_jitter > 0:
            bright = 1+(random.random()*2-1)*self.brightness_jitter
            image_enhancer = ImageEnhance.Brightness(image)
            label_enhancer = ImageEnhance.Brightness(label)
            image = image_enhancer.enhance(bright)
            label = label_enhancer.enhance(bright)
        if self.contrast_jitter > 0:
            contr = 1+(random.random()*2-1)*self.contrast_jitter
            image_enhancer = ImageEnhance.Contrast(image)
            label_enhancer = ImageEnhance.Contrast(label)
            image = image_enhancer.enhance(contr)
            label = label_enhancer.enhance(contr)
        if self.saturation_jitter > 0:
            sat = 1+(random.random()*2-1)*self.saturation_jitter
            image_enhancer = ImageEnhance.Color(image)
            label_enhancer = ImageEnhance.Color(label)
            image = image_enhancer.enhance(sat)
            label = label_enhancer.enhance(sat)
            
        # import matplotlib.pyplot as plt
        # plt.imshow(label)
        # plt.show()

        # Random crop
        if self.crop == 'random' and self.image_size != None:
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(self.image_size, self.image_size))
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)
        # Central crop
        if self.crop == 'center' and self.image_size != None:
            image = TF.center_crop(image, [self.image_size, self.image_size])
            label = TF.center_crop(label, [self.image_size, self.image_size])
        # Transform to tensor
        image = TF.to_tensor(image)
        label = TF.to_tensor(label)
        # image = torch.clip(image, min=0, max=1)
        # label = torch.clip(label, min=0, max=1)
        return image, label
