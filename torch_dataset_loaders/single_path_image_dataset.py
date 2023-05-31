import os

from .image_dataset import ImageDataset


class SinglePathImageDataset(ImageDataset):

    def __init__(self, DATASET_PATH, image_size, crop='random'):
        super().__init__(image_size, crop)
        self.image_paths = []
        for name in os.listdir(DATASET_PATH):
            path = os.path.join(DATASET_PATH, name)
            if os.path.isdir(path):
                for image_name in os.listdir(path):
                    self.image_paths.append(os.path.join(path, image_name))
            else:
                self.image_paths.append(os.path.join(DATASET_PATH, name))

    def __len__(self):
        return len(self.image_paths)