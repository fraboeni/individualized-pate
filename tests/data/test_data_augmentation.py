import numpy as np

from per_point_pate.data.data_augmentation import get_transform
import torch

def test_get_transform_mnist():
    
    # create data in numpy format, channels last
    n_data = 10000
    x = np.random.rand(n_data, 28, 28)

    # ToTensor transform results in channels first
    transform = get_transform('mnist')
    sample_transformed = transform(x[0])
    assert np.shape(sample_transformed) == (1, 28, 28)

    # same with additional data augmentation
    transform = get_transform('mnist', augmentation=True)
    sample_transformed = transform(x[0])
    assert np.shape(sample_transformed) == (1, 28, 28)

def test_get_transform_cifar10():
    
    # create data in numpy format, channels last
    n_data = 10000
    x = np.random.rand(n_data, 32, 32, 3)

    # ToTensor transform results in channels first
    transform = get_transform('cifar10')
    sample_transformed = transform(x[0])
    assert np.shape(sample_transformed) == (3, 32, 32)

    # same with additional data augmentation
    transform = get_transform('cifar10', augmentation=True)
    sample_transformed = transform(x[0])
    assert np.shape(sample_transformed) == (3, 32, 32)