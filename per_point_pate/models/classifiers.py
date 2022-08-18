from base64 import encode
from loguru import logger
import numpy as np
import os
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import time


def encode_one_hot(labels, n_classes):
    """
    This method transforms labels given by integers to arrays of zeroes and ones. E.g. out of four possible classes, the
    class '0' is encoded as [1, 0, 0, 0] while class '3' is encoded as [0, 0, 0, 1].
    @param labels:      array that contains labels to be one-hot-encoded
    @param n_classes:   int to specify the number of different classes
    @return:            array that contains one-hot-encoded labels
    """
    assert len(
        labels.shape
    ) == 1, f'labels has to be a 1d-array, but it has shape {labels.shape}'
    assert np.max(labels) < n_classes, f'There should only exist labels from 0 to {n_classes - 1}, but ' \
                                       f'{np.max(labels)} was given.'
    encoding = np.zeros((labels.size, n_classes), dtype=np.int8)
    encoding[np.arange(labels.size), labels] = 1
    return encoding


class Classifier:

    def __init__(self, input_size: tuple, n_classes: int):
        """
        This abstract class defines a type of ML model for classification tasks with all properties and methods
        relevant for this repository.
        @param input_size:  tuple to specify the form of inputs
        @param n_classes:   int to specify the number of different classes in the data distribution corresponding to the
                            concrete classifier
        """
        self.input_size = input_size
        self.n_classes = n_classes
        self.instance = None
        self.statistics = {}

    def build(self) -> object:
        """
        This method builds the classifier and prepares it for training.
        @return:    Classifier to be trained
        """
        raise NotImplementedError

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
            y_val: np.ndarray) -> object:
        """
        This method executes the training of the classifier.
        @param x_train: array that contains the features of all training data
        @param y_train: array that contains the labels of all training data
        @param x_val:   array that contains the features of all validation data
        @param y_val:   array that contains the labels of all validation data
        @return:        Classifier that was trained
        """
        raise NotImplementedError

    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        This method applies the classifier to the features of data points.
        @param x:   array that contains the features of data points
        @return:    array of probabilities for each class and for each given data point
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """
        This method stores the classifier at a given path.
        @param path:    string that specifies the location where the classifier will be stored
        @return:        None
        """
        raise NotImplementedError

    @staticmethod
    def load(path: str) -> object:
        """
        This method loads a classifier from pickle-files at the given location.
        @param path:    string that specifies the location of a stored classifier to be restored
        @return:        Classifier that was to be restored
        """
        raise NotImplementedError

    @staticmethod
    def confidences_to_classes(confidences: np.ndarray) -> np.ndarray:
        """
        This method transforms predictions that where given as probabilities of classes to classes.
        @param confidences: array that contains probabilities estimated by the classifier for features belonging to
                            classes
        @return:            array of predicted classes
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        This method applies the classifier to features of data points and transforms the resulting probabilities to
        classes.
        @param x:   array that contains features of data points
        @return:    array of predicted classes
        """
        return self.confidences_to_classes(self.eval(x))

    def accuracy(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        This method calculates the accuracy of the classifier on a given test set.
        @param x_test:  array that contains features of test data points
        @param y_test:  array that contains labels of test data points
        @return:        float of the achieved test accuracy
        """
        return round(sum(self.predict(x_test) == y_test) / len(y_test), 3)
