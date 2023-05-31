import random
import numpy as np
from torch.utils.data import Dataset

class MultipleDataset(Dataset):
    # gets a list of datasets, and randomly returns sample from one of them each time

    def __init__(self, datasets: list[Dataset]):
        super().__init__()
        self.datasets = datasets

    def __getitem__(self, idx):
        i = int(random.random()*self.datasets.__len__())  # int computes the floor
        return self.datasets[i].__getitem__(idx)

    def __len__(self):
        min = np.inf
        for dataset in self.datasets:
            if min > dataset.__len__():
                min = dataset.__len__()
        return min