from typing import Dict
import math
import numpy as np
import sklearn
from tqdm import tqdm

from per_point_pate.studies.parameters import ExperimentParameters, PATEParameters
from per_point_pate.models.pytorch.classifier_wrapper import ClassifierWrapper
from per_point_pate.privacy.pate import PATE, average_dp_budgets


def linear_costs(costs):
    res = []
    for cs in costs:
        r = []
        for c in cs:
            try:
                linear = round(math.exp(c), 3)
            except OverflowError:
                linear = float('inf')
            r.append(linear)
        res.append(r)
    return res


def main(
        prms: ExperimentParameters,
        aggregator: str,
        alphas: np.ndarray,
        public_data: np.array,
        budgets_per_sample: Dict,  # TODO: Is this a dict?
        mapping_t2p: Dict,  # TODO: Is this a dict?
):
    # unpack public data
    x_public_data, y_public_data = public_data

    # calculate predictions
    predictions = []
    model = ClassifierWrapper(input_size=np.shape(x_public_data)[1:],
                              architecture=prms.models.architecture,
                              dataset=prms.data.data_name,
                              n_classes=10,
                              seed=prms.pate.seed)
    for t in tqdm(range(prms.pate.n_teachers),
                  desc='calculate teacher predictions'):
        model = model.load(f'{prms.teachers_dir}/teacher_{t}')
        predictions.append(model.predict(x_public_data).numpy())
    predictions = np.array(predictions)

    # run pate algorithm to vote for labels
    pate = PATE(
        seed=prms.pate.seed,
        n_teachers=prms.pate.n_teachers,
        n_classes=10,
        predictions=predictions,
        budgets=budgets_per_sample,
        mapping=mapping_t2p,
        aggregator_name=aggregator,
        collector_name=prms.pate.collector,
        delta=prms.pate.delta,
        alphas=alphas,
        sigma=prms.pate.sigma,
        n_labels=prms.pate.n_labels,
        sigma1=prms.pate.sigma1,
        t=prms.pate.t,
    )
    pate.prepare()
    pate.predict_all()
    pate.simplify_budgets_costs()

    # label predictions with rejected votes being -1
    y_pred = np.array(pate.labels, dtype=np.longlong)
    n_votes = len(y_pred)

    filter_responds = y_pred != -1
    y_pred_clean = y_pred[filter_responds]
    true_labels = y_public_data[:n_votes][filter_responds]
    label_accuracy = sklearn.metrics.accuracy_score(y_true=true_labels,
                                                    y_pred=y_pred_clean)

    # features = pate.X[:n_votes]
    features = x_public_data[:n_votes]

    avg_budget = average_dp_budgets(epsilons=prms.pate.budgets,
                                    deltas=[prms.pate.delta] *
                                    len(prms.pate.budgets),
                                    weights=list(prms.pate.epsilon_weights))[0]

    # collect results
    # TODO: prms removed, log from caller
    # TODO: 'budgets_linear' removed, is it used at all?
    statistics = {
        'accuracy': label_accuracy,
        'n_votings': n_votes,
        'n_labels': len(y_pred_clean),
        'alpha_curve': str(pate.alpha_history),
        'ratios': str(pate.ratios),
        'avg_budget': round(avg_budget, 3),
        'avg_budget_linear': round(math.exp(avg_budget), 3),
        # TODO: how does 'epsilons' differ from prms.budgets?
        'simple_budgets': str([round(b, 3) for b in pate.simple_budgets]),
        'costs_curve': pate.simple_costs,
        'costs_curve_linear': str(linear_costs(pate.simple_costs)),
    }

    return features, y_pred, statistics
