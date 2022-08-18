from loguru import logger
import math
from time import time
import numpy as np
from typing import Tuple
from per_point_pate.models.pytorch.classifier_wrapper import ClassifierWrapper

from per_point_pate.studies.parameters import ExperimentParameters
from per_point_pate.privacy.pate import average_dp_budgets


def main(
    prms: ExperimentParameters,
    train_data: Tuple[np.array],
    test_data: Tuple[np.array],
):
    # upack data
    x_train, y_train = train_data
    x_test, y_test = test_data

    # build model
    model = ClassifierWrapper(input_size=np.shape(x_train)[1:],
                              architecture=prms.models.architecture,
                              dataset=prms.data.data_name,
                              n_classes=10)
    model.build()

    # train model
    batch_size = min(64, int(len(x_train) * 0.1))
    if batch_size < 16:
        logger.warning(f"Found extremely low batch size of {batch_size}.")

    train_start_time = time()
    train_loss_curve = model.fit(
        data_train=(x_train, y_train),
        batch_size=batch_size,
        n_epochs=prms.models.student_epochs,
        lr=prms.models.lr,
        weight_decay=prms.models.weight_decay,
    )
    train_time = time() - train_start_time

    test_accuracy, test_accuracy_by_class = model.accuracy(x_test=x_test,
                                                           y_test=y_test)
    test_precision, test_precision_by_class = model.precision(x_test=x_test,
                                                              y_test=y_test)
    test_recall, test_recall_by_class = model.recall(x_test=x_test,
                                                     y_test=y_test)

    avg_budget = average_dp_budgets(epsilons=prms.pate.budgets,
                                    deltas=[prms.pate.delta] *
                                    len(prms.pate.budgets),
                                    weights=list(prms.pate.epsilon_weights))[0]

    # collect statistics
    # TODO: prms removed, log from caller
    # TODO: 'costs', 'n_limit', 'n_labels', 'limit' removed, log from caller
    statistics = {
        'model_architecture': prms.models.architecture,
        'n_data_train': len(y_train),
        'avg_budget': round(avg_budget, 3),
        'avg_budget_linear': round(math.exp(avg_budget), 3),
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
    for key, value in model.__dict__.items():
        if key not in ['instance', 'statistics']:
            if type(value) not in [list, dict, tuple]:
                statistics[key] = value
            else:
                statistics[key] = str(value)

    for key, value in model.statistics.items():
        if type(value) not in [list, dict, tuple]:
            statistics[key] = value
        else:
            statistics[key] = str(value)

    return model, statistics
