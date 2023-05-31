from PIL import Image
from torchvision.transforms.functional import resize


class AutoResizer:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, image):
        return resize_if_needed(image, self.image_size)


def resize_if_needed(image: Image, image_size: int) -> Image:
    width, height = image.size
    if width < image_size or height < image_size:
        image = resize(image, [image_size, image_size])
    return image
