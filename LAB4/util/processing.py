from typing import List, Tuple

import torch.utils.data
import torchvision


def get_dataset(root_path: str, is_train: bool = True)\
        -> torch.utils.data.Dataset:
    return torchvision.datasets.MNIST(root=root_path,
                                      train=is_train,
                                      download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor()
                                      ]))


def get_datasets(root_path: str) -> List[torch.utils.data.Dataset]:
    return [get_dataset(root_path=root_path, is_train=x)
            for x in (True, False)]


def get_gan_dataset(root_path: str,
                    scale: Tuple[int, int] = (64, 64),
                    is_train: bool = True)\
        -> torch.utils.data.Dataset:
    return torchvision.datasets.MNIST(root=root_path,
                                      train=is_train,
                                      download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.Resize(scale),
                                          torchvision.transforms.ToTensor(),
                                      ]))


def get_gan_datasets(root_path: str, scale: Tuple[int, int] = (64, 64))\
        -> List[torch.utils.data.Dataset]:
    return [get_gan_dataset(root_path=root_path, scale=scale, is_train=x)
            for x in (True, False)]
