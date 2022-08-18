from pkgutil import get_data
from webbrowser import get
import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger
from typing import Union


class NumpyDataset(Dataset):

    def __init__(self,
                 features: np.array,
                 targets: Union[np.array, None] = None,
                 transform=None,
                 target_transform=None,
                 device: str = 'cuda'):
        if targets is not None:
            assert len(features) == len(targets)
        self.features = features
        self.targets = targets
        # self.features = torch.Tensor(features)
        # self.targets = torch.Tensor(targets)
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        features = self.features[idx]
        features = np.squeeze(features)
        if self.transform:
            features = self.transform(features)
            features = features.float()

        if self.targets is None:
            return features

        target = self.targets[idx]
        if self.target_transform:
            target = self.target_transform(target)

        return features, target
