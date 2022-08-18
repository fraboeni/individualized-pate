from matplotlib.transforms import Transform
import numpy as np
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Union


def load_split(
    data_subdir: Path,
    split: Union[bool, str],
    dataset_class: Dataset,
):

    if type(split) == bool:
        dataset = dataset_class(root=data_subdir, train=split, download=True)
    elif type(split) == str:
        dataset = dataset_class(root=data_subdir, split=split, download=True)
    else:
        raise RuntimeError(f"Got unknown split type {split}")

    x = dataset.data
    if dataset_class == datasets.SVHN:
        y = dataset.labels
    else:
        y = dataset.targets

    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    return x, y


def load_dataset(
    data_subdir: Path,
    dataset_class: Dataset,
):
    """Loads dataset, and concatenates train and test sets

    Args:
        data_subdir (Path): _description_
        dataset_class (Dataset): _description_

    Returns:
        _type_: _description_
    """
    path_x = data_subdir / 'x.npy'
    path_y = data_subdir / 'y.npy'

    if path_x.is_file() and path_y.is_file():
        x = np.load(path_x)
        y = np.load(path_y)
        return x, y

    x_train, y_train = load_split(data_subdir, True, dataset_class)
    x_test, y_test = load_split(data_subdir, False, dataset_class)

    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    np.save(path_x, x)
    np.save(path_y, y)

    return x, y


def load_mnist(data_dir: Path, ):
    (x, y) = load_dataset(
        data_subdir=data_dir / 'mnist',
        dataset_class=datasets.MNIST,
    )
    shape = np.shape(x)[1:]
    classes = 10

    return (x, y), shape, classes


def load_fashion_mnist(data_dir: Path, ):
    (x, y) = load_dataset(
        data_subdir=data_dir / 'fashion_mnist',
        dataset_class=datasets.FashionMNIST,
    )
    shape = np.shape(x)[1:]
    classes = 10

    return (x, y), shape, classes


def load_cifar10(data_dir: Path, ):
    (x, y) = load_dataset(
        data_subdir=data_dir / 'cifar10',
        dataset_class=datasets.CIFAR10,
    )
    shape = np.shape(x)[1:]
    classes = 10

    return (x, y), shape, classes


def load_svhn(data_dir: Path, ):
    data_subdir = data_dir / 'svhn'

    path_x = data_subdir / 'x.npy'
    path_y = data_subdir / 'y.npy'

    if path_x.is_file() and path_y.is_file():
        x = np.load(path_x)
        y = np.load(path_y)

    else:
        x_train, y_train = load_split(data_subdir, 'train', datasets.SVHN)
        x_test, y_test = load_split(data_subdir, 'test', datasets.SVHN)

        x = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test])

        x = np.moveaxis(x, 1, -1)

        np.save(path_x, x)
        np.save(path_y, y)

    shape = np.shape(x)[1:]
    classes = 10

    return (x, y), shape, classes
