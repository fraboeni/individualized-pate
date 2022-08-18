from loguru import logger
import numpy as np
import pickle
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm

from per_point_pate.models.pytorch.pytorch_model import PytorchModel, PytorchModelArgs
from per_point_pate.data.data_augmentation import get_transform
from per_point_pate.models.pytorch.training import accuracy_by_class, evaluate, evaluate_precision, evaluate_recall, train_epoch, accuracy
from per_point_pate.models.classifiers import Classifier
from per_point_pate.data.numpy_dataset import NumpyDataset


class ClassifierWrapper(Classifier):
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        architecture: str,
        dataset: str,
        n_classes: int,
        cuda: bool = True,
        seed: int = 0,
    ):
        super().__init__(
            input_size=input_size,
            n_classes=n_classes,
        )
        self.args = PytorchModelArgs(
            architecture=architecture,
            dataset=dataset,
            num_classes=n_classes,
            cuda=cuda,
        )
        self.seed = seed

    def build(self):
        """
        This method builds the classifier and prepares it for training.
        @param seed: Random seed set for initializing parameters with torch.
        """
        torch.manual_seed(self.seed)
        self.instance = PytorchModel(self.args)
        if self.args.cuda:
            self.instance.cuda()

    def fit(self,
            data_train: Tuple[np.ndarray, np.ndarray],
            batch_size,
            n_epochs,
            lr,
            weight_decay,
            data_val: Tuple[np.ndarray, np.ndarray] = None):
        """
        This method executes the training of the classifier.
        @param x_train: array that contains the features of all training data
        @param y_train: array that contains the labels of all training data
        @param x_val:   array that contains the features of all validation data
        @param y_val:   array that contains the labels of all validation data
        @return:        Classifier that was trained
        """
        torch.manual_seed(self.seed)

        x_train, y_train = data_train
        p = np.random.permutation(np.arange(len(y_train)))
        x_train = x_train[p]
        y_train = y_train[p]

        ds_train = NumpyDataset(features=x_train,
                                targets=y_train,
                                transform=get_transform(
                                    dataset=self.args.dataset,
                                    augmentation=True))
        # loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
        loader_train = DataLoader(ds_train,
                                  batch_size=batch_size,
                                  shuffle=True)

        if data_val:
            x_val, y_val = data_val
            ds_val = NumpyDataset(
                features=x_val,
                targets=y_val,
                transform=get_transform(dataset=self.args.dataset),
            )
            loader_val = DataLoader(ds_val,
                                    batch_size=batch_size,
                                    num_workers=4)

        optimizer = torch.optim.Adam(params=self.instance.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                      factor=0.1,
                                      verbose=True)

        train_loss_curve, val_loss_curve = [], []
        self.instance.train()
        epochs = tqdm(range(n_epochs))
        for epoch in epochs:
            train_loss = train_epoch(self.instance, loader_train, optimizer,
                                     self.args)
            scheduler.step(train_loss)
            train_loss_curve.append(train_loss)
            desc = f'train_loss: {train_loss}'

            if data_val:
                val_loss, val_acc = accuracy(self.instance, loader_val,
                                             self.args)
                val_loss_curve.append(val_loss)
                desc = f'{desc}, val_loss: {val_loss}, val_acc: {val_acc}'

            epochs.set_description(desc=desc)

        return train_loss_curve

    def eval(self, x: np.ndarray, batch_size: int = None):
        """
        This method applies the convolutional neural network to the features of given data points.
        @param x:   array that contains the features of data points
        @return:    array of probabilities for each class and for each given data point
        """
        ds_test = NumpyDataset(
            features=x,
            transform=get_transform(self.args.dataset),
        )
        if batch_size is None:
            batch_size = len(x)
        loader_test = DataLoader(ds_test, batch_size=batch_size)
        predictions = evaluate(self.instance, loader_test, self.args)
        return predictions

    @staticmethod
    def confidences_to_classes(confidences: np.ndarray) -> np.ndarray:
        _, classes = torch.nn.functional.softmax(confidences).topk(1)
        classes = classes.view(-1).long()
        return classes

    def precision(self,
                  x_test: np.ndarray,
                  y_test: np.ndarray,
                  gpus=1) -> float:
        """
        This method calculates the precision of the classifier on a given test set.
        @param x_test:  array that contains features of test data points
        @param y_test:  array that contains labels of test data points
        @return:        float of the achieved test accuracy
        """
        ds_test = NumpyDataset(
            features=x_test,
            targets=y_test,
            transform=get_transform(self.args.dataset),
        )
        loader_test = DataLoader(ds_test, batch_size=1)
        precision, by_class = evaluate_precision(self.instance, loader_test,
                                                 self.args)
        return precision, by_class

    def recall(self, x_test: np.ndarray, y_test: np.ndarray, gpus=1) -> float:
        """
        This method calculates the recall of the classifier on a given test set.
        @param x_test:  array that contains features of test data points
        @param y_test:  array that contains labels of test data points
        @return:        float of the achieved test accuracy
        """
        ds_test = NumpyDataset(
            features=x_test,
            targets=y_test,
            transform=get_transform(self.args.dataset),
        )
        loader_test = DataLoader(ds_test, batch_size=1)
        precision, by_class = evaluate_recall(self.instance, loader_test,
                                              self.args)
        return precision, by_class

    def accuracy(self,
                 x_test: np.ndarray,
                 y_test: np.ndarray,
                 gpus=1) -> float:
        """
        This method calculates the accuracy of the classifier on a given test set.
        @param x_test:  array that contains features of test data points
        @param y_test:  array that contains labels of test data points
        @return:        float of the achieved test accuracy
        """
        ds_test = NumpyDataset(
            features=x_test,
            targets=y_test,
            transform=get_transform(self.args.dataset),
        )
        loader_test = DataLoader(ds_test, batch_size=1)
        test_loss, test_acc = accuracy(self.instance, loader_test, self.args)
        by_class = accuracy_by_class(self.instance, loader_test, self.args)

        return test_acc, by_class

    def save(self, path):
        """
        This method stores the convolutional neural network at a given path.
        @param path:    string that specifies the location where the classifier will be stored
        @return:        None
        """
        torch.save(self.instance.state_dict(), f'{path}/model.pt')
        # TODO: clean up workaround to omit instance from pickle
        self.instance = None
        with open(file=f'{path}/model.pickle', mode='wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        This method loads a convolutional neural network from pickle-files at the given location.
        @param path:    string that specifies the location of a stored classifier to be restored
        @return:        Classifier that was restored
        """
        with open(file=f'{path}/model.pickle', mode='rb') as f:
            model = pickle.load(f)
        model.instance = PytorchModel(model.args)
        model.instance.load_state_dict(torch.load(f'{path}/model.pt'))
        if model.args.cuda:
            model.instance = model.instance.cuda()

        return model
