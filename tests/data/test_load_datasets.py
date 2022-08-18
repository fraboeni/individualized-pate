from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from per_point_pate.data.load_datasets import load_cifar10, load_fashion_mnist, load_mnist, load_svhn
import pytest


def test_load_mnist():
    with TemporaryDirectory(prefix='tests_per_point_pate') as d:
        (x, y), shape, n_classes = load_mnist(data_dir=Path(d))
        assert np.shape(x) == (70000, 28, 28)
        assert np.shape(y) == (70000,)
        assert shape == (28, 28)
        assert n_classes == 10

def test_load_fashion_mnist():
    with TemporaryDirectory(prefix='tests_per_point_pate') as d:
        (x, y), shape, n_classes = load_fashion_mnist(data_dir=Path(d))
        assert np.shape(x) == (70000, 28, 28)
        assert np.shape(y) == (70000,)
        assert shape == (28, 28)
        assert n_classes == 10

def test_load_cifar10():
    with TemporaryDirectory(prefix='tests_per_point_pate') as d:
        (x, y), shape, n_classes = load_cifar10(data_dir=Path(d))
        assert np.shape(x) == (60000, 32, 32, 3)
        assert np.shape(y) == (60000,)
        assert shape == (32, 32, 3)
        assert n_classes == 10

@pytest.mark.slow
def test_load_svhn():
    with TemporaryDirectory(prefix='tests_per_point_pate') as d:
        (x, y), shape, n_classes = load_svhn(data_dir=Path(d))
        assert np.shape(x) == (630420, 32, 32, 3)
        assert np.shape(y) == (630420,)
        assert shape == (32, 32, 3)
        assert n_classes == 10