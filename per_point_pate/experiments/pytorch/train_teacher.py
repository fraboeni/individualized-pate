from time import time
from loguru import logger
import numpy as np
from typing import Tuple

from per_point_pate.models.pytorch.classifier_wrapper import ClassifierWrapper
from per_point_pate.studies.parameters import ModelParameters


def main(
    model_prms: ModelParameters,
    # TODO: replace data_name, models should not need it
    data_name: str,
    train_data: Tuple[np.array],
    test_data: Tuple[np.array],
    seed: int,
):
    # set random seed
    np.random.seed(seed)

    # unpack data
    x_train, y_train = train_data
    x_test, y_test = test_data

    # build model
    model = ClassifierWrapper(
        input_size=np.shape(x_train)[1:],
        architecture=model_prms.architecture,
        dataset=data_name,
        n_classes=10,
        seed=seed,
    )
    model.build()

    # train model
    batch_size = min(64, int(len(x_train) * 0.1))
    if batch_size < 16:
        logger.warning(f"Found extremely low batch size of {batch_size}.")

    train_start_time = time()
    train_loss_curve = model.fit(
        data_train=(x_train, y_train),
        batch_size=batch_size,
        n_epochs=model_prms.teacher_epochs,
        lr=model_prms.lr,
        weight_decay=model_prms.weight_decay,
    )
    train_time = time() - train_start_time

    test_accuracy, test_accuracy_by_class = model.accuracy(x_test=x_test,
                                                           y_test=y_test)
    test_precision, test_precision_by_class = model.precision(x_test=x_test,
                                                              y_test=y_test)
    test_recall, test_recall_by_class = model.recall(x_test=x_test,
                                                     y_test=y_test)

    statistics = {
        'model': model_prms.architecture,
        'train_time': train_time,
        'test_accuracy': round(test_accuracy, 3),
        'test_accuracy_by_class':
        [round(a, 3) for a in test_accuracy_by_class],
        'test_precision': round(test_precision, 3),
        'test_precision_by_class':
        [round(a, 3) for a in test_precision_by_class],
        'test_recall': round(test_recall, 3),
        'test_recall_by_class': [round(a, 3) for a in test_recall_by_class],
        'train_loss_curve': train_loss_curve,
    }

    return model, statistics
