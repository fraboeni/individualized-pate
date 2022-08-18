import numpy as np
from pathlib import Path
import torch
from torch import nn

from per_point_pate.models.pytorch.classifier_wrapper import ClassifierWrapper

def test_classfier_wrapper_load():
    path = Path(__file__).parent / 'resources'
    model = ClassifierWrapper.load(path)
    assert isinstance(model, ClassifierWrapper)
    assert isinstance(model.instance, nn.Module)

def test_classifier_wrapper_evaluate():
    path = Path(__file__).parent / 'resources'
    model = ClassifierWrapper.load(path)
    n_data = 1000
    x = np.random.rand(n_data, 28, 28)
    y = np.random.randint(
        low=0,
        high=10,
        size=n_data
    )

    predictions = model.eval(x)
    print(predictions.size())
    assert isinstance(predictions, torch.Tensor)

def test_classifier_wrapper_accuracy():
    path = Path(__file__).parent / 'resources'
    model = ClassifierWrapper.load(path)
    n_data = 100
    x = np.random.rand(n_data, 28, 28)
    y = np.random.randint(
        low=0,
        high=10,
        size=n_data
    )

    accuracy, by_class = model.accuracy(x, y)
    assert isinstance(accuracy, float)
    assert isinstance(by_class, np.ndarray)

def test_classifier_wrapper_precision():
    path = Path(__file__).parent / 'resources'
    model = ClassifierWrapper.load(path)
    n_data = 100
    x = np.random.rand(n_data, 28, 28)
    y = np.random.randint(
        low=0,
        high=10,
        size=n_data
    )

    precision, by_class = model.precision(x, y)
    assert isinstance(precision, float)
    assert isinstance(by_class, np.ndarray)

def test_classifier_wrapper_recall():
    path = Path(__file__).parent / 'resources'
    model = ClassifierWrapper.load(path)
    n_data = 100
    x = np.random.rand(n_data, 28, 28)
    y = np.random.randint(
        low=0,
        high=10,
        size=n_data
    )

    recall, by_class = model.recall(x, y)
    assert isinstance(recall, float)
    assert isinstance(by_class, np.ndarray)